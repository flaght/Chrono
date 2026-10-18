import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def ki025(close, volume, openint, window, weriod, ewm=False):
    """
    超跌反弹信号因子 (Oversold Rebound Signal)

    基于FSD超跌反弹理论改进，识别超跌后的反弹机会

    原理:
        1. 获利盘比例 < 10% 表示极度超跌
        2. 持仓量增加 + 价格企稳 = 多头建仓
        3. 成交量萎缩 + 获利盘触底 = 抛压减弱

    信号条件 (期货改进版):
        - 获利盘比例处于低位 (< 20%分位)
        - 持仓量相对增加 (表示有新多头入场)
        - 价格止跌企稳 (波动率收窄)

    参数:
        close: 收盘价 DataFrame (分钟级)
        volume: 成交量 DataFrame (分钟级)
        openint: 持仓量 DataFrame (分钟级)
        window: 外层平滑窗口
        weriod: 日内周期 (如240分钟/天)
        ewm: 是否使用指数加权

    返回:
        超跌反弹信号强度 (越高表示反弹概率越大)

    信号解读:
        > 0.7: 强烈超跌反弹信号
        0.3-0.7: 中等反弹可能
        < 0.3: 无明显信号
    """
    method = 'ewm' if ewm else 'rolling'

    # === 1. 计算获利盘比例 ===
    daily_volume = roller_sum(volume, weriod, weriod, method)
    daily_openint = roller_mean(openint, weriod, weriod, method)

    turnover_rate = daily_volume / (daily_openint + 1e-10)
    turnover_rate = turnover_rate.clip(upper=1.0)

    price_std = roller_std(close, weriod, weriod, method)
    ema_span = window * 2
    cost_center = roller_mean(close, ema_span, ema_span, method)

    price_deviation = (close - cost_center) / (price_std + 1e-10)
    profit_ratio = 1 / (1 + np.exp(-price_deviation))

    # === 2. 获利盘处于低位信号 ===
    # 计算获利盘的历史分位数
    profit_rank = roller_rank(profit_ratio, window * 5, window * 5, 'rolling', pct=True)
    # 获利盘越低，超跌程度越高
    oversold_signal = 1 - profit_rank

    # === 3. 持仓量增加信号 (多头建仓) ===
    oi_change = openint.diff(weriod)
    oi_change_norm = oi_change / (roller_std(oi_change, weriod, weriod, method) + 1e-10)
    # sigmoid映射到0-1
    oi_increase_signal = 1 / (1 + np.exp(-oi_change_norm))

    # === 4. 波动率收窄信号 (止跌企稳) ===
    volatility = roller_std(close.pct_change(), weriod, weriod, method)
    vol_rank = roller_rank(volatility, window * 5, window * 5, 'rolling', pct=True)
    # 波动率越低，企稳程度越高
    stable_signal = 1 - vol_rank

    # === 5. 综合信号 ===
    # 超跌(0.5) + 持仓增加(0.3) + 企稳(0.2)
    combined_signal = (
        0.5 * oversold_signal +
        0.3 * oi_increase_signal +
        0.2 * stable_signal
    )

    # 最终平滑
    alpha = roller_mean(combined_signal, window, window, method)

    return alpha

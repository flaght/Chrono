import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def ki024(close, volume, openint, window, weriod, ewm=False):
    """
    获利盘变化率因子 (Profit Ratio Change)

    捕捉获利盘比例的变化速度，用于识别趋势加速或反转

    原理:
        1. 基于ky001计算获利盘比例
        2. 计算获利盘比例的变化率
        3. 获利盘快速上升 = 趋势加速
        4. 获利盘快速下降 = 趋势减弱

    信号解读:
        ch_profit > 0.3: 获利盘快速增加，多头趋势强化
        ch_profit < -0.3: 获利盘快速减少，空头趋势或止损出场
        |ch_profit| < 0.1: 横盘整理

    参数:
        close: 收盘价 DataFrame (分钟级)
        volume: 成交量 DataFrame (分钟级)
        openint: 持仓量 DataFrame (分钟级)
        window: 外层平滑窗口
        weriod: 日内周期 (如240分钟/天)
        ewm: 是否使用指数加权

    返回:
        获利盘变化率因子值
    """
    method = 'ewm' if ewm else 'rolling'

    # 计算日内累计成交量
    daily_volume = roller_sum(volume, weriod, weriod, method)

    # 计算日内平均持仓量
    daily_openint = roller_mean(openint, weriod, weriod, method)

    # 估算换手率
    turnover_rate = daily_volume / (daily_openint + 1e-10)
    turnover_rate = turnover_rate.clip(upper=1.0)

    # 计算价格波动率
    price_std = roller_std(close, weriod, weriod, method)

    # 计算持仓成本中枢
    ema_span = window * 2
    cost_center = roller_mean(close, ema_span, ema_span, method)

    # 计算价格相对成本中枢的偏离
    price_deviation = (close - cost_center) / (price_std + 1e-10)

    # 获利盘比例
    profit_ratio = 1 / (1 + np.exp(-price_deviation))

    # 计算获利盘变化率 (相对于weriod周期前)
    profit_ratio_lag = profit_ratio.shift(weriod)
    ch_profit = profit_ratio - profit_ratio_lag

    # 平滑处理
    alpha = roller_mean(ch_profit, window, window, method)

    return alpha

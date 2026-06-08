import numpy as np
from lumina.impulse.fixed import *


def ki002(close, volume, window, weriod, select_ratio=0.2, ewm=False):
    """
    信息分布涨跌幅因子 (URet)

    基于信息分布均匀程度识别反转效应，选取信息分布最不均匀的交易日收益构建因子

    原理:
        1. 信息分布均匀程度用成交量的变异系数衡量: Z = std(volume) / mean(volume)
        2. Z值越大，信息分布越不均匀，股价更容易反应过度，表现为反转
        3. Z值越小，信息分布越均匀，股价更容易反应不足，表现为动量
        4. 选取Z值最大(信息分布最不均匀)的交易日收益，构建反转因子

    参数:
        close: 收盘价 DataFrame (分钟级)
        volume: 成交量 DataFrame (分钟级)
        window: 外层平滑窗口
        weriod: 日内周期 (如240分钟/天)
        select_ratio: 选取比例 (默认0.2，即选Z值最大的20%的天数)
        ewm: 是否使用指数加权

    返回:
        信息分布涨跌幅因子值
    """
    method = 'ewm' if ewm else 'rolling'

    # 计算日收益率
    daily_return = close.pct_change()

    # 计算日内成交量的变异系数 (信息分布均匀程度的代理变量)
    # Z = std(volume) / mean(volume)
    vol_std = roller_std(volume, weriod, weriod, method)
    vol_mean = roller_mean(volume, weriod, weriod, method)

    # 变异系数 Z
    z_score = vol_std / (vol_mean + 1e-10)

    # 将日收益率按Z值加权
    lookback = 20 * weriod  # 回看约20天
    min_periods = lookback // 2

    # 计算Z值的排名分位数 (0-1)
    z_rank = roller_rank(z_score, lookback, min_periods, 'rolling', pct=True)

    # 只保留Z值排名在top select_ratio的日子的收益
    threshold = 1 - select_ratio
    high_z_mask = (z_rank >= threshold).astype(float)

    # 加权平均收益 (只计算高Z值日子)
    weighted_ret = daily_return * high_z_mask
    weight_sum = roller_sum(high_z_mask, lookback, min_periods, method)

    # 计算平均收益
    core1 = roller_sum(weighted_ret, lookback, min_periods, method) / (weight_sum + 1e-10)

    # 最终用 window 做平滑
    alpha = roller_mean(core1, window, window, method)

    return alpha

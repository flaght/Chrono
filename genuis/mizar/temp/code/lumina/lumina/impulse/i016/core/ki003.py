import numpy as np
from lumina.impulse.fixed import *


def ki003(close, volume, window, weriod, select_ratio=0.2, ewm=False):
    """
    信息分布涨跌幅因子差分版 (URet')

    取信息分布最不均匀与最均匀的交易日收益之差

    原理:
        URet' = RetPart5 - RetPart1
        RetPart5: 信息分布最不均匀的天数的收益均值
        RetPart1: 信息分布最均匀的天数的收益均值

    参数:
        close: 收盘价 DataFrame (分钟级)
        volume: 成交量 DataFrame (分钟级)
        window: 外层平滑窗口
        weriod: 日内周期 (如240分钟/天)
        select_ratio: 两端选取比例 (默认0.2)
        ewm: 是否使用指数加权

    返回:
        信息分布涨跌幅因子差分值
    """
    method = 'ewm' if ewm else 'rolling'

    # 计算日收益率
    daily_return = close.pct_change()

    # 计算日内成交量的变异系数
    vol_std = roller_std(volume, weriod, weriod, method)
    vol_mean = roller_mean(volume, weriod, weriod, method)
    z_score = vol_std / (vol_mean + 1e-10)

    # 回看窗口
    lookback = 20 * weriod
    min_periods = lookback // 2

    # Z值排名
    z_rank = roller_rank(z_score, lookback, min_periods, 'rolling', pct=True)

    # 高Z值 (信息分布不均匀) 的收益
    high_threshold = 1 - select_ratio
    high_z_mask = (z_rank >= high_threshold).astype(float)
    high_z_ret = daily_return * high_z_mask
    high_z_weight = roller_sum(high_z_mask, lookback, min_periods, method)
    ret_part5 = roller_sum(high_z_ret, lookback, min_periods, method) / (high_z_weight + 1e-10)

    # 低Z值 (信息分布均匀) 的收益
    low_threshold = select_ratio
    low_z_mask = (z_rank <= low_threshold).astype(float)
    low_z_ret = daily_return * low_z_mask
    low_z_weight = roller_sum(low_z_mask, lookback, min_periods, method)
    ret_part1 = roller_sum(low_z_ret, lookback, min_periods, method) / (low_z_weight + 1e-10)

    # 差分
    core1 = ret_part5 - ret_part1

    # 最终用 window 做平滑
    alpha = roller_mean(core1, window, window, method)

    return alpha

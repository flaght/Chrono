import numpy as np
from lumina.impulse.fixed import *


def ki001(close, volume, openint, window, weriod, ewm=False):
    """
    CPV因子 - 修正持仓量价量相关性

    基于修正后的持仓量计算价量相关系数，捕捉多空信号

    原理:
        1. 原始日内持仓量呈"山谷"形态(T+0交易者行为导致)
        2. 通过修正将持仓量形态从"山谷"变为"山峰"
        3. 计算价格变化与修正持仓量变化的相关系数

    修正公式:
        ΔOI(T+1)_i = (ΔV_i / ΔV) * ΔOI  (T+1交易者持仓量变化)
        ΔOI(T+0)_i = -1 * [ΔOI_i - ΔOI(T+1)_i]  (T+0交易者持仓量变化)
        OI_corrected = cumsum(ΔOI(T+0) + ΔOI(T+1))

    参数:
        close: 收盘价 DataFrame (分钟级)
        volume: 成交量 DataFrame (分钟级)
        openint: 持仓量 DataFrame (分钟级)
        window: 外层平滑窗口
        weriod: 日内周期 (如240分钟/天)
        ewm: 是否使用指数加权

    返回:
        CPV因子值

    信号解读:
        PV > 0: 看多信号
        PV < 0: 看空信号
    """
    method = 'ewm' if ewm else 'rolling'

    # 计算价格变化
    delta_price = close.diff()

    # 计算持仓量变化
    delta_oi = openint.diff()

    # 计算成交量变化 (用于加权)
    delta_volume = volume.diff().clip(lower=0)

    # 计算日内总持仓量变化
    total_delta_oi = roller_sum(delta_oi, weriod, weriod, method)

    # 计算日内总成交量
    total_volume = roller_sum(delta_volume, weriod, weriod, method)

    # 计算T+1交易者的持仓量变化权重
    volume_weight = delta_volume / (total_volume + 1e-10)

    # 计算T+1交易者的持仓量变化
    delta_oi_t1 = volume_weight * total_delta_oi

    # 计算T+0交易者的持仓量变化 (乘-1修正)
    delta_oi_t0 = -1 * (delta_oi - delta_oi_t1)

    # 修正后的总持仓量变化
    delta_oi_corrected = delta_oi_t0 + delta_oi_t1

    # 计算价格变化与修正持仓量变化的滚动相关系数
    core1 = roller_corr(delta_price, delta_oi_corrected, weriod, weriod, method)

    # 最终用 window 做平滑
    alpha = roller_mean(core1, window, window, method)

    return alpha

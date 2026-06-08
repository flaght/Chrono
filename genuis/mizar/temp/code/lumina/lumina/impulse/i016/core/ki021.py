# -*- encoding:utf-8 -*-
"""
量幅同向因子 (华西证券)
来源: 基于量价因子的ETF组合策略.pdf

公式: correlation(Volume变化率, 振幅变化率)
含义: 成交量变化与价格振幅变化的正相关性
      量幅同向：放量时振幅大，缩量时振幅小 → 健康的价格运动
"""
from lumina.impulse.fixed import *


def ki021(high, low, volume, window, weriod, ewm=False):
    """
    量幅同向因子

    参数:
        high: 最高价序列
        low: 最低价序列
        volume: 成交量序列
        window: 外层平滑窗口
        weriod: 回望周期
        ewm: 是否使用指数加权

    返回:
        alpha: 因子值
    """
    method = 'ewm' if ewm else 'rolling'

    # 计算成交量变化率
    vol_change = volume / volume.shift(1) - 1

    # 计算日内振幅变化率
    amplitude = high / low - 1
    amp_change = amplitude / amplitude.shift(1) - 1

    # 计算量变与幅变的相关系数
    core = roller_corr(vol_change, amp_change, weriod, weriod, method)

    # 外层平滑
    alpha = roller_mean(core, window, window, method)

    return alpha

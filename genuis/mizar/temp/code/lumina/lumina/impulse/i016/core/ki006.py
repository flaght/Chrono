"""
因子: dz001 - 收盘价与统计均值价差因子
来源: 东证期货 - 国债期货量价因子挖掘 (2022-07-12)
"""
from lumina.impulse.fixed import *


def ki006(close, window, weriod, ewm=False):
    """
    收盘价与统计均值价差因子

    研报表达式: sub(X1, X8) 或 sub(X1, X18)
    表现: 夏普1.65, 年化11.4-16.4%
    """
    method = 'ewm' if ewm else 'rolling'

    mean_price = roller_mean(close, weriod, weriod, method)
    spread = close - mean_price

    # 标准化
    spread_std = roller_std(spread, weriod, weriod, method)
    spread_norm = safe_div(spread, spread_std + 1e-8)

    # 最终平滑
    alpha = roller_mean(spread_norm, window, window, method)

    return -alpha

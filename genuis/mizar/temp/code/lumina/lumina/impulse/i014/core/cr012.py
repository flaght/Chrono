"""
cr012：N期最高价与最低价的极差与收盘价波动率复合因子，衡量极端波动与风险。
计算方式：先计算N期最高-最低极差，再与N期收盘价标准差相乘，最后做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr012(high, low, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    #price_range = high.rolling(weriod).max() - low.rolling(weriod).min()
    price_range = roller_max(high, window=weriod,
                                 min_periods=1) - roller_min(
                                     low, window=weriod, min_periods=1)
    #vol = close.rolling(weriod).std()
    vol = roller_std(close, weriod, 1, method)
    factor = price_range * vol
    factor = roller_mean(factor, window, 1, method)
    return factor

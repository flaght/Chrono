"""
cr014：N日收盘价收益率的偏度与最高价极差复合因子，衡量收益分布偏斜与高点波动。
计算方式：先计算N日收盘价对数收益率的偏度，再与N日最高价极差相乘，最后做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr014(close, high, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    log_ret = np.log(close / close.shift(1))
    #skew = log_ret.rolling(weriod).skew()
    skew = roller_skew(log_ret, weriod, 1, method)
    #high_range = high.rolling(weriod).max() - high.rolling(weriod).min()
    high_range = roller_max(high, window=weriod,
                                min_periods=1) - roller_min(
                                    high, window=weriod, min_periods=1)
    factor = skew * high_range
    factor = roller_mean(factor, window, 1, method)
    return factor

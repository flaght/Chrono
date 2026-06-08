"""
cr015：N日收盘价与开盘价的对数收益率与最高价极差的相关系数复合因子，衡量收益与高点波动的同步性。
计算方式：先计算N日收盘-开盘对数收益率与最高价极差的相关系数，再做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr015(close, open, high, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    log_ret = np.log(close / open)
    #high_range = high.rolling(weriod).max() - high.rolling(weriod).min()
    high_range = roller_max(high, window=weriod,
                                min_periods=1) - roller_min(
                                    high, window=weriod, min_periods=1)
    #corr = log_ret.rolling(weriod).corr(high_range)
    corr = roller_corr(log_ret, high_range, weriod, 1, method)
    factor = roller_mean(corr, window, 1, method)
    return factor

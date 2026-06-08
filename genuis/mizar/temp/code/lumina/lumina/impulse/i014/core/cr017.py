"""
cr017：N日收盘价与开盘价的对数收益率与最高价极差的协方差复合因子，衡量收益与高点波动的联动性。
计算方式：先计算N日收盘-开盘对数收益率与最高价极差的协方差，再做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr017(close, open, high, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    log_ret = np.log(close / open)
    #high_range = high.rolling(weriod).max() - high.rolling(weriod).min()
    high_range = roller_max(high, window=weriod,
                                min_periods=1) - roller_min(
                                    high, window=weriod, min_periods=1)
    #cov = log_ret.rolling(weriod).cov(high_range)
    cov = roller_cov(log_ret, high_range, weriod, 1, method)
    factor = roller_mean(cov, window, 1, method)
    return factor

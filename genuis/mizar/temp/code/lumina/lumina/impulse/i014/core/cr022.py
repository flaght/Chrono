"""
cr022：N期最高价与最低价极差的偏度与收盘价波动率的协方差复合因子，衡量极端波动分布与风险的联动性。
计算方式：先计算N期最高-最低极差的偏度与N期收盘价标准差的协方差，再做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr022(high, low, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    #price_range = high.rolling(weriod).max() - low.rolling(weriod).min()
    price_range = roller_max(high, window=weriod,
                                 min_periods=1) - roller_min(
                                     low, window=weriod, min_periods=1)
    #skew = price_range.rolling(weriod).skew()
    skew = roller_skew(price_range, weriod, 1, method)
    #vol = close.rolling(weriod).std()
    vol = roller_std(close, weriod, 1, method)
    #cov = skew.rolling(weriod).cov(vol)
    cov = roller_cov(skew, vol, weriod, 1, method)
    factor = roller_mean(cov, window, 1, method)
    return factor

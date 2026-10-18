"""
cr026：N期最高价与最低价极差的自相关系数与收盘价波动率的协方差复合因子，衡量极端波动惯性与风险的联动性。
计算方式：先计算N期最高-最低极差的自相关系数与N期收盘价标准差的协方差，再做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr026(high, low, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    #price_range = high.rolling(weriod).max() - low.rolling(weriod).min()
    price_range = roller_max(high, window=weriod,
                                 min_periods=1) - roller_min(
                                     low, window=weriod, min_periods=1)
    autocorr = price_range.rolling(weriod).apply(
        lambda x: pd.Series(x).autocorr(), raw=False)
    vol = roller_std(close, weriod, 1, method)
    #vol = close.rolling(weriod).std()
    cov = roller_cov(autocorr, vol, weriod, 1, method)
    #cov = autocorr.rolling(weriod).cov(vol)
    factor = roller_mean(cov, window, 1, method)
    return factor

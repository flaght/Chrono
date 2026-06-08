"""
cr013：N日收盘价与开盘价的对数收益率偏度与成交量波动率复合因子，衡量收益分布偏斜与量能风险。
计算方式：先计算N日收盘-开盘对数收益率的偏度，再与N日成交量标准差相乘，最后做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr013(close, open, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    log_ret = np.log(close / open)
    #skew = log_ret.rolling(weriod).skew()
    #skew = roller_skew(log_ret, weriod, 1, method)
    skew = roller_skew(log_ret, weriod, 1, method)
    #vol_std = volume.rolling(weriod).std()
    vol_std = roller_std(volume, weriod, 1, method)
    factor = skew * vol_std
    factor = roller_mean(factor, window, 1, method)
    return factor

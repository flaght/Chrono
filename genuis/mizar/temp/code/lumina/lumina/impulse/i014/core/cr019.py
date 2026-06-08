"""
cr019：N期收盘价与开盘价的对数收益率与成交量波动率的相关系数复合因子，衡量收益与量能风险的同步性。
计算方式：先计算N期收盘-开盘对数收益率与N期成交量标准差的相关系数，再做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr019(close, open, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    log_ret = np.log(close / open)
    #vol_std = volume.rolling(weriod).std()
    vol_std = roller_std(volume, weriod, 1, method)
    #corr = log_ret.rolling(weriod).corr(vol_std)
    corr = roller_corr(log_ret, vol_std, weriod, 1, method)
    factor = roller_mean(corr, window, 1, method)
    return factor

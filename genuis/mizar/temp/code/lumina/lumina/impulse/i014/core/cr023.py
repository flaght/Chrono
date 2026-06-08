"""
cr023：N期收盘价对数收益率的峰度与成交量波动率的相关系数复合因子，衡量收益分布陡峭与量能风险的同步性。
计算方式：先计算N期收盘价对数收益率的峰度与N期成交量标准差的相关系数，再做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr023(close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    log_ret = np.log(close / close.shift(1))
    #kurt = log_ret.rolling(weriod).kurt()
    kurt = roller_kurt(log_ret, weriod, 1, 'rolling')
    #vol_std = volume.rolling(weriod).std()
    vol_std = roller_std(volume, weriod, 1, method)
    #corr = kurt.rolling(weriod).corr(vol_std)
    corr = roller_corr(kurt, vol_std, weriod, 1, method)
    factor = roller_mean(corr, window, 1, method)
    return factor

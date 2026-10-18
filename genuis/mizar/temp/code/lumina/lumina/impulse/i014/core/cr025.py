"""
cr025：N期收盘价对数收益率的自相关系数与成交量变化率的相关系数复合因子，衡量收益惯性与量能变化的同步性。
计算方式：先计算N期收盘价对数收益率的自相关系数与N期成交量变化率的相关系数，再做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr025(close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    log_ret = np.log(close / close.shift(1))
    autocorr = log_ret.rolling(weriod).apply(lambda x: pd.Series(x).autocorr(),
                                             raw=False)
    vol_chg = volume.pct_change()
    #corr = autocorr.rolling(weriod).corr(vol_chg)
    corr = roller_corr(autocorr, vol_chg, weriod, 1, method)
    factor = roller_mean(corr, window, 1, method)
    return factor

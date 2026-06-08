"""
cr011：N期收盘价与开盘价的对数收益率与成交量变化率的协方差复合因子，衡量价量联动的波动性。
计算方式：先计算N期收盘-开盘对数收益率与成交量变化率的协方差，再做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr011(close, open, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    log_ret = np.log(close / open)
    vol_chg = volume.pct_change()
    #cov = log_ret.rolling(weriod).cov(vol_chg)
    cov = roller_cov(log_ret, vol_chg, weriod, 1, method)
    factor = roller_mean(cov, window, 1, method)
    return factor

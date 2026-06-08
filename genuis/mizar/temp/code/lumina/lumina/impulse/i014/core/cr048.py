"""
cr048：N期收盘价与开盘价的对数收益率偏度与持仓量(openint)波动率复合因子，衡量收益分布偏斜与持仓风险。
计算方式：先计算N期收盘-开盘对数收益率的偏度，再与N期持仓量(openint)标准差相乘，最后做滑动平均。
本因子为cr013的持仓量(openint)版本。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *

def cr048(close, open, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    log_ret = np.log(close / open)
    skew = roller_skew(log_ret, weriod, 1, method)
    oi_std = roller_std(openint, weriod, 1, method)
    factor = skew * oi_std
    factor = roller_mean(factor, window, 1, method)
    return factor 
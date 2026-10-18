"""
cr049：N期收盘价与开盘价的对数收益率与持仓量(openint)波动率的相关系数复合因子，衡量收益与持仓风险的同步性。
计算方式：先计算N期收盘-开盘对数收益率与N期持仓量(openint)标准差的相关系数，再做滑动平均。
本因子为cr019的持仓量(openint)版本。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *

def cr049(close, open, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    log_ret = np.log(close / open)
    oi_std = roller_std(openint, weriod, 1, method)
    corr = roller_corr(log_ret, oi_std, weriod, 1, method)
    factor = roller_mean(corr, window, 1, method)
    return factor 
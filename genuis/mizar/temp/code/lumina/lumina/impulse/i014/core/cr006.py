"""
cr006：N期最高价与收盘价的相关系数因子，衡量价格高点与收盘的同步性。
计算方式：计算N期内最高价与收盘价的相关系数，做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr006(high, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    corr1 = roller_corr(high, close, weriod, 1, method)
    factor = roller_mean(corr1, window, 1, method)
    return factor

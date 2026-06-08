"""
cr009：N日收盘价与成交量的相关性与极端值复合因子，衡量价量共振与极端波动。
计算方式：先计算N日收盘价与成交量的相关系数，再与N日收盘价的极端涨跌幅相乘，最后做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr009(close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    corr = close.rolling(weriod).corr(volume)
    extreme = (close.pct_change(weriod).abs()
               > close.pct_change(weriod).abs().rolling(weriod).quantile(0.95)
               ).astype(float)
    factor = corr * extreme
    factor = roller_mean(factor, window, 1, method)
    return factor

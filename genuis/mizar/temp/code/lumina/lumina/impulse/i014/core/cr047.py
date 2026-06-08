"""
cr047：N期收盘价与持仓量(openint)的相关性与极端值复合因子，衡量价持仓共振与极端波动。
计算方式：先计算N期收盘价与持仓量(openint)的相关系数，再与N期收盘价的极端涨跌幅相乘，最后做滑动平均。
本因子为cr009的持仓量(openint)版本。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *

def cr047(close, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    corr = close.rolling(weriod).corr(openint)
    extreme = (close.pct_change(weriod).abs()
               > close.pct_change(weriod).abs().rolling(weriod).quantile(0.95)
               ).astype(float)
    factor = corr * extreme
    factor = roller_mean(factor, window, 1, method)
    return factor 
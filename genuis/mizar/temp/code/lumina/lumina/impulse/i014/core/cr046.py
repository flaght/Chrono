"""
cr046：N期最高价与最低价区间突破与持仓量(openint)变化复合因子，衡量价格突破与持仓量配合。
计算方式：先计算N期最高价与最低价的区间突破幅度，再与N期持仓量(openint)变化率相乘，最后做滑动平均。
本因子为cr008的持仓量(openint)版本。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *

def cr046(high, low, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    range_break = (roller_max(high, weriod, 1, 'rolling') -
                   roller_min(low, weriod, 1, 'rolling')
                   ) / roller_min(low, weriod, 1, 'rolling')
    oi_chg = openint.pct_change(weriod)
    factor = range_break * oi_chg
    factor = roller_mean(factor, window, 1, method)
    return factor 
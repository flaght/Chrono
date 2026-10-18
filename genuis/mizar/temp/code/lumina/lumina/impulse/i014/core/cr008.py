"""
cr008：N期最高价与最低价区间突破与成交量变化复合因子，衡量价格突破与量能配合。
计算方式：先计算N期最高价与最低价的区间突破幅度，再与N期成交量变化率相乘，最后做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr008(high, low, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    range_break = (roller_max(high, weriod, 1, 'rolling') -
                   roller_min(low, weriod, 1, 'rolling')
                   ) / roller_min(low, weriod, 1, 'rolling')
    vol_chg = volume.pct_change(weriod)
    factor = range_break * vol_chg
    factor = roller_mean(factor, window, 1, method)
    return factor

"""
cr020：N期最高价与最低价极差与成交量变化率的协方差复合因子，衡量极端波动与量能变化的联动性。
计算方式：先计算N期最高-最低极差与N期成交量变化率的协方差，再做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr020(high, low, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    #price_range = high.rolling(weriod).max() - low.rolling(weriod).min()
    price_range = roller_max(high, window=weriod,
                                 min_periods=1) - roller_min(
                                     low, window=weriod, min_periods=1)
    vol_chg = volume.pct_change()
    #cov = price_range.rolling(weriod).cov(vol_chg)
    cov = roller_cov(price_range, vol_chg, weriod, 1, method)
    factor = roller_mean(cov, window, 1, method)
    return factor

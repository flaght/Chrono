"""
cr044：高低价极差的自适应分位阈值触发因子，衡量极端行情爆发概率。
计算方式：N期极差大于N期分位阈值时输出1，否则为0，滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr044(high, low, threshold, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    price_range = roller_max(high, window=weriod,
                                 min_periods=1) - roller_min(
                                     low, window=weriod, min_periods=1)
    quantile_val = roller_quantile(price_range, threshold, weriod, 1, 'rolling')
    trigger = (price_range > quantile_val).astype(float)
    factor = roller_mean(trigger, window, 1, method)
    return factor

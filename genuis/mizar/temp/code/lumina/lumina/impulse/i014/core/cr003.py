"""
cr003：N期最高价与最低价的对数比率波动因子，衡量价格区间的波动幅度。
计算方式：取N期内最高价与最低价的对数之差，做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr003(high, low, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    log_hl = np.log(
        roller_max(high, weriod, 1, 'rolling') /
        roller_min(low, weriod, 1, 'rolling'))
    factor = roller_mean(log_hl, window, 1, method)
    return factor

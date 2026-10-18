"""
cr051：持仓量(openint)极差的自适应分位阈值触发因子，衡量持仓极端变化爆发概率。
计算方式：N期持仓量极差大于N期分位阈值时输出1，否则为0，滑动平均。
本因子为cr044的持仓量(openint)版本。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *

def cr051(openint, threshold, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    oi_range = roller_max(openint, weriod, 1, 'rolling') - roller_min(openint, weriod, 1, 'rolling')
    quantile_val = roller_quantile(oi_range, threshold, weriod, 1, 'rolling')
    trigger = (oi_range > quantile_val).astype(float)
    factor = roller_mean(trigger, window, 1, method)
    return factor 
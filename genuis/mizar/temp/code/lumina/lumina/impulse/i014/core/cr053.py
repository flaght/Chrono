"""
cr053：短期与长期持仓量(openint)波动率比值的sigmoid变换因子，衡量持仓波动率聚集与极端变化。
计算方式：N1期与N2期持仓量标准差之比，经过sigmoid变换后滑动平均。
本因子为cr041的持仓量(openint)版本。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *

def cr053(openint, slow, fast, window, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    short_std = roller_std(openint, fast, 1, method)
    long_std = roller_std(openint, slow, 1, method)
    ratio = short_std / (long_std + 1e-8)
    sig = 1 / (1 + np.exp(-ratio))
    factor = roller_mean(sig, window, 1, method)
    return factor 
"""
cr041：短期与长期波动率比值的sigmoid变换因子，衡量波动率聚集与极端变化。
计算方式：N1期与N2期收盘价对数收益率标准差之比，经过sigmoid变换后滑动平均。
"""
import pdb
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr041(close, slow, fast, window, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = np.log(close / close.shift(1))
    short_std = roller_std(rets, fast, 1, method)
    long_std = roller_std(rets, slow, 1, method)
    ratio = short_std / (long_std + 1e-8)
    sig = 1 / (1 + np.exp(-ratio))
    factor = roller_mean(sig, window, 1, method)
    return factor

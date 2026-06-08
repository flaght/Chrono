import pdb
import pandas as pd
from lumina.impulse.fixed import *


## 一致买入交易
def dv005(close, open, high, low, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    if_grow = np.where(close > open, 1, 0)
    if_con = np.where(np.abs(close - open) <= 0.5 * (high - low), 1, 0)

    alpha = roller_sum(volume * if_grow * if_con, weriod, 1,
                       method) / roller_sum(volume, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)

    return alpha

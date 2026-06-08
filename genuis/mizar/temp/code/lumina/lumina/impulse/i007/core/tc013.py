import pdb
from lumina.impulse.fixed import *


# PGO
def tc013(high, low, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    pgo = close - roller_mean(close, weriod, 1, method)

    high_low_range = high - low
    high_close_range = (high - close.shift()).abs()
    low_close_range = (low - close.shift()).abs()

    true_range = high_low_range.copy()

    cond1 = true_range < high_close_range
    true_range[cond1] = high_close_range[cond1]
    cond2 = true_range < low_close_range
    true_range[cond2] = low_close_range[cond2]
    atr = roller_mean(true_range, weriod, weriod, method)

    pgo = pgo / atr

    alpha = roller_mean(pgo, window, 1, method)
    return alpha

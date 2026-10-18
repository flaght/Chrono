import pdb
from lumina.impulse.fixed import *


def tc003(high, low, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    high_low_range = high - low
    high_low_range = high - low
    high_close_range = (high - close.shift()).abs()
    low_close_range = (low - close.shift()).abs()
    cond1 = high_low_range > high_close_range
    high_close_range[cond1] = high_low_range[cond1]
    cond2 = high_close_range > low_close_range
    low_close_range[cond2] = high_close_range[cond2]

    tr = low_close_range
    atr = roller_mean(tr, weriod, weriod, method)
    ret = close - close.shift(weriod) + 0.0001
    alpha = 2 * ret / (atr + atr.shift(weriod))

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

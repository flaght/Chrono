import pdb
from lumina.impulse.fixed import *


# cksp
def tv016(high, low, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    high_low_range = high - low
    high_close_range = (high - close.shift()).abs()
    low_close_range = (low - close.shift()).abs()

    true_range = high_low_range.copy()

    cond1 = true_range < high_close_range
    true_range[cond1] = high_close_range[cond1]
    cond2 = true_range < low_close_range
    true_range[cond2] = low_close_range[cond2]
    atr = roller_mean(true_range, weriod, weriod, method)

    long_stop = roller_max(high, weriod, 1, 'rolling') - 3 * atr
    long_stop = roller_max(long_stop, window, 1, 'rolling')

    short_stop = roller_min(low, weriod, 1, 'rolling') + 3 * atr
    short_stop = roller_min(short_stop, window, 1, 'rolling')

    cksp = long_stop - short_stop

    alpha = roller_mean(cksp, window, 1, method)

    return alpha
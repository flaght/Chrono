import pdb
from lumina.impulse.fixed import *


## chop
def tv015(high, low, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    diff = roller_max(high, weriod, 1, 'rolling') - roller_min(
        low, weriod, 1, 'rolling') + 1e-5

    high_low_range = high - low
    high_close_range = (high - close.shift()).abs()
    low_close_range = (low - close.shift()).abs()

    true_range = high_low_range.copy()

    cond1 = true_range < high_close_range
    true_range[cond1] = high_close_range[cond1]
    cond2 = true_range < low_close_range
    true_range[cond2] = low_close_range[cond2]
    atr = roller_mean(true_range, weriod, weriod, method)

    atr_sum = roller_sum(atr, window, 1, method)

    chop = 10 * (np.log(atr_sum) - np.log(diff)) / np.log(weriod)

    alpha = roller_mean(chop, window, 1, method)

    return alpha

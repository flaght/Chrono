import pdb
from lumina.impulse.fixed import *


## vortex
def tv018(high, low, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    high_low_range = high - low
    high_close_range = (high - close.shift()).abs()
    low_close_range = (low - close.shift()).abs()

    true_range = high_low_range.copy()

    cond1 = true_range < high_close_range
    true_range[cond1] = high_close_range[cond1]
    cond2 = true_range < low_close_range
    true_range[cond2] = low_close_range[cond2]

    tr_sum = roller_sum(true_range, weriod, 1, method)

    vmp = (high - low.shift(1)).abs()
    vmm = (low - high.shift(1)).abs()

    vip = roller_sum(vmp, weriod, 1, method) / tr_sum
    vim = roller_sum(vmm, weriod, 1, method) / tr_sum

    vortex = vip / vim

    alpha = roller_mean(vortex, window, 1, method)
    return alpha

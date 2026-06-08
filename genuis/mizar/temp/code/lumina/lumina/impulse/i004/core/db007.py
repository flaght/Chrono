import pdb
from lumina.impulse.fixed import *

# factor_dongfang_core_2
def db007(open, high, low, close, window, weriod, ewm=False):

    method = 'ewm' if ewm else 'rolling'

    twap = (high + low + close + open) / 4

    mean_twap = roller_mean(twap, weriod, weriod, method)
    max_high = roller_max(high, weriod, weriod, 'rolling')
    min_low = roller_min(low, weriod, weriod, 'rolling')

    mean_twap = roller_mean(mean_twap, weriod, weriod, method)
    max_high = roller_max(max_high, weriod, weriod, 'rolling')
    min_low = roller_min(min_low, weriod, weriod, 'rolling')

    core1 = ((mean_twap - min_low) / (max_high - mean_twap)) + 1e-4

    alpha = roller_mean(core1, window, window, method)

    return alpha
import pdb
from lumina.impulse.fixed import *

## massi
def tv011(high, low, window, fast, slow, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    high_low_range = high - low
    hl_ema1 = roller_mean(high_low_range, fast, 1, method)
    hl_ema2 = roller_mean(hl_ema1, slow, 1, method)

    hl_ratio = hl_ema1 / hl_ema2

    alpha = roller_mean(hl_ratio, window, 1, method)
    return alpha
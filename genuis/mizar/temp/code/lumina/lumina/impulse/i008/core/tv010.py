import pdb
from lumina.impulse.fixed import *


## KC
def tv010(high, low, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    high_low_range = high - low
    true_range = high_low_range.copy()
    true_range[true_range < high - close.shift(1)] = high - close.shift(1)
    true_range[true_range < close.shift(1) - low] = close.shift(1) - low

    basis = roller_mean(close, weriod, 1, method)
    band = roller_mean(true_range, weriod, 1, method)
    lower = basis - 2 * band
    upper = basis + 2 * band

    alpha = upper / lower

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

import pdb
from lumina.impulse.fixed import *

# ad
def tv001(high, low, close, volume, window, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    ad = 2 * close - high - low
    high_low_range = high - low
    ad *= volume / high_low_range
    alpha = roller_mean(ad, 1, 1, method)

    alpha = roller_mean(alpha, window, 1, method)

    return alpha

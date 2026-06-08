import pdb
from lumina.impulse.fixed import *

#adosc
def tv002(high, low, close, volume, window, fast, slow, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    ad = 2 * close - high - low
    high_low_range = high - low
    ad *= volume / high_low_range
    ad = roller_mean(ad, 1, 1, method)

    fast_ad = roller_mean(ad, fast, 1, method)
    slow_ad = roller_mean(ad, slow, 1, method)

    adosc = fast_ad - slow_ad

    alpha = roller_mean(adosc, window, 1, method)

    return alpha

from lumina.impulse.fixed import *


# ad
def oi031(high, low, close, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    ad = 2 * close - high - low
    high_low_range = high - low
    ad *= openint / high_low_range
    alpha = roller_mean(ad, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)

    return alpha

import pdb
from lumina.impulse.fixed import *

# cmf
def tv003(high, low, close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    ad = 2 * close - high - low

    ad *= volume / (high - low)
    cmf = roller_mean(ad, weriod, 1, method) / roller_mean(volume, weriod, 1, method)

    alpha = roller_mean(cmf, window, 1, method)
    return alpha

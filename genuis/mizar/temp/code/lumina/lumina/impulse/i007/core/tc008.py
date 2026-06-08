import pdb
from lumina.impulse.fixed import *

# CCI
def tc008(high, low, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    c = 0.015
    typical_price = (high + low + close) / 3.0
    mean_typical_price = roller_mean(typical_price, weriod, 1, method)
    median_typical_price = roller_median(typical_price, weriod, 1, 'rolling')
    cci = typical_price - mean_typical_price
    cci = cci / (c * median_typical_price)

    alpha = roller_mean(cci, window, 1, method)
    return alpha

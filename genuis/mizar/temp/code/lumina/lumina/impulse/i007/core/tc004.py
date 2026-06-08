import pdb
from lumina.impulse.fixed import *


# AO
def tc004(high, low, window, fast, slow, ewm=False):
    if slow < fast:
        fast, slow = slow, fast
    method = 'ewm' if ewm else 'rolling'
    median_price = (high + low) / 2
    fast_sma = roller_mean(median_price, fast, 1, method)
    slow_sma = roller_mean(median_price, slow, 1, method)
    alpha = fast_sma - slow_sma

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

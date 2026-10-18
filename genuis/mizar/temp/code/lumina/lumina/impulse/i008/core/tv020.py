import pdb
from lumina.impulse.fixed import *

# DEMA
def tv020(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    ema1 = roller_mean(close, weriod, 1, method)
    ema2 = roller_mean(ema1, weriod, 1, method)
    dema = 2 * ema1 - ema2

    alpha = roller_mean(dema, window, 1, method)
    return dema
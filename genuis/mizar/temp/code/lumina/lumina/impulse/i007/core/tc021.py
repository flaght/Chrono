import pdb
from lumina.impulse.fixed import *


# vwma
def tc021(close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    pv = close * volume
    vwma = roller_mean(pv, weriod, 1, method) / roller_mean(
        volume, weriod, 1, method)
    alpha = roller_mean(vwma, window, 1, method)
    return alpha

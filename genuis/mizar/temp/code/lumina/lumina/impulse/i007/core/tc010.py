import pdb
from lumina.impulse.fixed import *

# eri
def tc010(high, low, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    ma1 = roller_mean(close, weriod, 1, method)
    bull = high - ma1
    bear = low - ma1
    eri = bull / bear

    alpha = roller_mean(eri, window, 1, method)
    return  alpha
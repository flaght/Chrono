import pdb
from lumina.impulse.fixed import *


# dpo
def tv017(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    ma = roller_mean(close, weriod, 1, method)
    dpo = close - ma

    alpha = roller_mean(dpo, window, 1, method)
    return alpha

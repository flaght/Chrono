import pdb
from lumina.impulse.fixed import *


# trima
def tc020(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    half_length = round(0.5 * (weriod + 1))
    sma1 = roller_mean(close, half_length, 1, method)
    sma2 = roller_mean(sma1, half_length, 1, method)

    alpha = roller_mean(sma2, window, 1, method)
    return alpha

import pdb
from lumina.impulse.fixed import *


# thermo
def tv013(high, low, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    thermoL = (low.shift(1) - low).abs()
    thermoH = (high - high.shift(1)).abs()

    thermo = thermoH.copy()
    thermo[thermo < thermoL] = thermoL[thermo < thermoL]

    alpha = roller_mean(thermo, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

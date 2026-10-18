import pdb
from lumina.impulse.fixed import *


# chkbar + clkbar
def tc007(close, high, low, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chkbar = roller_mean(((close - high) / high), weriod, 1, method)
    clkbar = roller_mean(((close - low) / low), weriod, 1, method)

    alpha = roller_mean(chkbar / clkbar, window, 1, method)
    return alpha

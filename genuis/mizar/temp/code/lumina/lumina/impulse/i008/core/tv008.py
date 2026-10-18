import pdb
from lumina.impulse.fixed import *


## bbands
def tv008(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    derviation = 2.0 * roller_std(close, weriod, 1, method)
    mid = roller_mean(close, weriod, 1, method)
    lower = mid - derviation
    upper = mid + derviation

    alpha = upper / lower

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

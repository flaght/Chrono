import pdb
from lumina.impulse.fixed import *

# alpha151
def iv006(value, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_log(close)
    buffer0 = value / roller_sum(value, weriod, 1, method)
    buffer1 = chg * buffer0
    buffer2 = roller_mean(buffer1, weriod, 1, method)

    factors = roller_sum(buffer1 * buffer2, weriod, 1, method)

    alpha = roller_mean(factors, window, 1, method)
    return alpha

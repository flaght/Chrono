import pdb
from lumina.impulse.fixed import *


# APO
def tc005(close, window, fast, slow, ewm=False):
    if slow < fast:
        fast, slow = slow, fast

    method = 'ewm' if ewm else 'rolling'
    fast_ema = roller_mean(close, fast, 1, method)
    slow_ema = roller_mean(close, slow, 1, method)
    alpha = fast_ema - slow_ema
    alpha = roller_mean(alpha, window, 1, method)
    return alpha

import pdb
from lumina.impulse.fixed import *


# PVO
def tc016(volume, window, fast, slow, scalar=None, ewm=False):
    if slow < fast:
        fast, slow = slow, fast
    method = 'ewm' if ewm else 'rolling'
    scalar = float(scalar) if scalar and scalar > 0 else 100000

    fast_ema = roller_mean(volume, fast, 1, method)
    slow_ema = roller_mean(volume, slow, 1, method)
    pvo = scalar * (fast_ema - slow_ema) / slow_ema

    alpha = roller_mean(pvo, window, 1, method)
    return alpha

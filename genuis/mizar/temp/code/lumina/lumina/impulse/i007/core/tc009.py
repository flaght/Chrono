import pdb
from lumina.impulse.fixed import *


## coppock
def tc009(close, window, weriod, fast, slow, scalar=None, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    scalar = float(scalar) if scalar and scalar > 0 else 100

    roc = (scalar * close.diff(fast) /
           close.shift(fast)) + (scalar * close.diff(slow) / close.shift(slow))
    alpha = roller_mean(roc, weriod, 1, method)
    
    alpha = roller_mean(alpha, window, 1, method)

    return alpha

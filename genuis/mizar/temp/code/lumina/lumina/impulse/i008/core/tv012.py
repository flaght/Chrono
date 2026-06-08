import pdb
from lumina.impulse.fixed import *

## pdist

def tv012(open, high, low, close, window, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    pdist = 2 * (high - low)
    pdist += (open - close.shift(1)).abs()
    pdist -= (close -open).abs()

    alpha = roller_mean(pdist, window, 1, method)
    return alpha
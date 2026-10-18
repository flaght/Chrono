import pdb
from lumina.impulse.fixed import *


## PSL
def tc015(close, open, window, weriod, scalar=None, ewm=False):
    scalar = float(scalar) if scalar and scalar > 0 else 10000
    method = 'ewm' if ewm else 'rolling'
    diff = np.sign(close - open)
    diff.fillna(0, inplace=True)
    diff[diff < 0] = 0
    psl = scalar * roller_sum(diff, weriod, 1, method) / weriod

    alpha = roller_mean(psl, window, 1, method)
    return alpha

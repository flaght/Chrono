import pdb
from lumina.impulse.fixed import *

# nvi
def tv006(close, volume, window, weriod, scalar=None, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    scalar = float(scalar) if scalar and scalar > 0 else 100

    mom = close.diff(weriod)
    roc = scalar * mom  / close.shift(weriod)
    sign = np.sign(volume.diff(1))
    nvi = sign[sign < 0].abs() * roc
    nvi = roller_sum(nvi, window, 1, method)

    nvi = roller_mean(nvi, window, 1, method)
    return nvi

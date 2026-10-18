import pdb
from lumina.impulse.fixed import *

## hma
def tv021(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    half_weriod = int(weriod / 2)
    sqrt_weriod = int(np.sqrt(weriod))

    wmaf = roller_mean(close, half_weriod, 1, method)
    wmas = roller_mean(close, weriod, 1, method)
    hma = roller_mean(2 * wmaf - wmas, sqrt_weriod, 1, method)

    alpha = roller_mean(hma, window, 1, method)
    return alpha
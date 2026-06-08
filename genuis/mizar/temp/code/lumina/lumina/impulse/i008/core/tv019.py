import pdb
from lumina.impulse.fixed import *


## VHF
def tv019(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    hcp = roller_max(close, weriod, 1, 'rolling')
    lcp = roller_min(close, weriod, 1, 'rolling')
    diff = np.fabs(close.diff(1))
    whf = np.fabs(hcp - lcp) / roller_sum(diff, weriod, 1, method)

    alpha = roller_mean(whf, window, 1, method)

    return alpha

import pdb
from lumina.impulse.fixed import *


def rv013(close, volume, threshold, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_log(close)
    vwar = roller_sum(chg * volume, weriod, 1, method) / roller_sum(
        volume, weriod, 1, method)

    wvwar = -roller_quantile(vwar, threshold, weriod, 1, 'rolling')
    
    factor = -vwar.mask(vwar >= wvwar, np.nan)

    factor1 = roller_mean(factor, weriod, 1, method)
    factor2 = factor - factor1

    alpha1 = roller_mean(factor1, window, 1, method)
    alpha2 = roller_mean(factor2, window, 1, method)
    return alpha1, alpha2

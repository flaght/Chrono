import pdb
from lumina.impulse.fixed import *


def rv011(close, threshold, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_log(close)
    factor = roller_quantile(chg, threshold, weriod, 1, 'rolling')

    factor1 = roller_mean(factor, weriod, 1, method)
    factor2 = factor - factor1

    alpha1 = roller_mean(factor1, window, 1, method)
    alpha2 = roller_mean(factor2, window, 1, method)
    return alpha1, alpha2

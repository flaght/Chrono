import pdb
from lumina.impulse.fixed import *
from lumina.impulse.i009.core.base import calc_umr

#kurt umr


def iv003(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_log(close)
    kurt = roller_kurt(chg, weriod, weriod, 'rolling')
    factor = calc_umr(kurt, weriod)

    alpha = roller_mean(factor, window, 1, method)
    return alpha

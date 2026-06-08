import pdb
from lumina.impulse.fixed import *
from lumina.impulse.i009.core.base import calc_umr


# std umr
def iv001(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_log(close)
    std = roller_std(chg, weriod, 1, method)
    factor = calc_umr(std, weriod)

    alpha = roller_mean(factor, window, 1, method)
    return alpha

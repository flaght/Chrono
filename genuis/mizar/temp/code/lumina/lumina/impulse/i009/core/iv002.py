import pdb
from lumina.impulse.fixed import *
from lumina.impulse.i009.core.base import calc_umr


# skew_umr
def iv002(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_log(close)
    skew = roller_skew(chg, weriod, 1, method)
    factor = calc_umr(skew, weriod)

    alpha = roller_mean(factor, window, 1, method)
    return alpha

import pdb
from lumina.impulse.fixed import *


# alpha154
def iv004(close, fast, slow, weriod, window, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_logx(close, fast, slow)
    factors = roller_sum(chg, weriod, 1, method)

    alpha = roller_mean(factors, window, 1, method)
    return alpha

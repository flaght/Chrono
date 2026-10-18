import pdb
from lumina.impulse.fixed import *


#route
def ixy014(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)
    rets_sum = roller_sum(rets, weriod, 1, method)
    rets_mean = roller_mean(rets, weriod, 1, method)
    alpha = rets_sum / rets_mean.abs()

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

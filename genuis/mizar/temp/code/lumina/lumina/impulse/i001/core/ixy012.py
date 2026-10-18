import pdb
from lumina.impulse.fixed import *


#maxdown.py
def ixy012(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)
    rets_sum = roller_sum(rets, weriod, 1, method)
    maxdd = roller_max(rets_sum, weriod, 1, 'rolling').expanding().max()
    alpha = rets_sum - maxdd

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

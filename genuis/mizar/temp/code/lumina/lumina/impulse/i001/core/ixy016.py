import pdb
from lumina.impulse.fixed import *


#skew
def ixy016(close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)
    turn_volume = volume
    rets_skew = roller_skew(rets, weriod, 1, method)
    volume_skew = roller_skew(turn_volume, weriod, 1, method)
    alpha = roller_corr(rets_skew, volume_skew, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

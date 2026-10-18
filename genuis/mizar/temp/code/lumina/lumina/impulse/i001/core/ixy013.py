import pdb
import pandas as pd
from lumina.impulse.fixed import *


#pvcorr
def ixy013(close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)
    turn_volume = volume / 1e6
    alpha = roller_corr(rets, turn_volume, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

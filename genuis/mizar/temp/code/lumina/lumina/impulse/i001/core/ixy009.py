import pdb
import pandas as pd
from lumina.impulse.fixed import *


#kurt
def ixy009(close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)
    turn_volume = volume / 1e6
    rets_kurt = roller_kurt(rets, weriod, 1, 'rolling')
    volume_kurt = roller_kurt(turn_volume, weriod, 1, 'rolling')
    alpha = roller_corr(rets_kurt, volume_kurt, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

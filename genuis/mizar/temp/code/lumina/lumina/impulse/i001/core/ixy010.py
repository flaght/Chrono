import pdb
import numpy as np
from lumina.impulse.fixed import *


#liquid1
def ixy010(volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    turn_volume = volume / 1e6
    turn_volume = safe_log(turn_volume, 1)
    turn_volume[turn_volume <= 0] = np.nan
    alpha = roller_mean(turn_volume, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)

    return alpha

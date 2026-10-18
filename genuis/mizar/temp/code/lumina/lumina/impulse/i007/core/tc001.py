import pdb
from lumina.impulse.fixed import *


def tc001(volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    vol = volume.copy()
    vol[vol <= 0] = np.nan
    log_vol = safe_log(vol)
    alpha = roller_mean(log_vol, weriod, weriod, method)
    alpha = roller_std(alpha, window, window, method)
    return alpha
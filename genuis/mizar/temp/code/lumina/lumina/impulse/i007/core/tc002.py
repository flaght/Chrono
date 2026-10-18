import pdb
from lumina.impulse.fixed import *


def tc002(volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    vol = volume.copy()
    vol[vol <= 0] = np.nan
    log_vol = safe_log(vol)

    alpha = roller_std(log_vol, weriod, 1, method)
    alpha = roller_mean(alpha, window, 1, method)
    return alpha

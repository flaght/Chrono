from lumina.impulse.fixed import *


def oi029(openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    oni = openint.copy()
    oni[oni <= 0] = np.nan
    log_oni = safe_log(oni)
    alpha = roller_mean(log_oni, weriod, weriod, method)
    alpha = roller_std(alpha, window, window, method)
    return alpha

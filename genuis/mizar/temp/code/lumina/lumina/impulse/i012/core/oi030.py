from lumina.impulse.fixed import *



def oi030(openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    oni = openint.copy()
    oni[oni <= 0] = np.nan
    log_oni = safe_log(oni)

    alpha = roller_std(log_oni, weriod, 1, method)
    alpha = roller_mean(alpha, window, 1, method)
    return alpha

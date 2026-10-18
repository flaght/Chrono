from lumina.impulse.fixed import *


#pvcorr
def oi008(close, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)
    turn_openint = openint / 1e6
    alpha = roller_corr(rets, turn_openint, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

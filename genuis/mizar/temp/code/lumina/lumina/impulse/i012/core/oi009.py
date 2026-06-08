from lumina.impulse.fixed import *


#skew
def oi009(close, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)
    turn_openint = openint / 1e6
    rets_skew = roller_skew(rets, weriod, 1, method)
    openint_skew = roller_skew(turn_openint, weriod, 1, method)
    alpha = roller_corr(rets_skew, openint_skew, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

from lumina.impulse.fixed import *


#kurt
def oi005(close, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)
    turn_openint = openint / 1e6
    rets_kurt = roller_kurt(rets, weriod, 1, 'rolling')
    openint_kurt = roller_kurt(openint, weriod, 1, 'rolling')
    alpha = roller_corr(rets_kurt, openint_kurt, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

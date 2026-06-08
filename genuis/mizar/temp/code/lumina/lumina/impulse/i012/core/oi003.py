from lumina.impulse.fixed import *


#illiq1
def oi003(close, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)
    turn_openint = openint / 1e6

    illiq = safe_div(rets, turn_openint)
    alpha = roller_mean(illiq, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

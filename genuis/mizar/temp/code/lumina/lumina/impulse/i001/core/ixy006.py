from lumina.impulse.fixed import *


#illiq1
def ixy006(close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)
    turn_volume = volume

    illiq = safe_div(rets, turn_volume)
    alpha = roller_mean(illiq, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

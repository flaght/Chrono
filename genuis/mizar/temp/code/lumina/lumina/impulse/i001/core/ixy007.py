from lumina.impulse.fixed import *


#illiq2
def ixy007(close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    rets = safe_log(close, 1)
    turn_volume = volume / 1e6
    rets_std = roller_mean(rets, weriod, 1, method)
    volume_std = roller_mean(turn_volume, weriod, 1, method)
    alpha = safe_div(rets_std, volume_std)

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

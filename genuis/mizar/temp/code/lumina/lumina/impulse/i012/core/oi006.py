from lumina.impulse.fixed import *


#liquid1
def oi006(openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    turn_openint = openint / 1e6
    turn_openint = safe_log(turn_openint, 1)
    turn_openint[turn_openint <= 0] = np.nan
    alpha = roller_mean(turn_openint, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)

    return alpha

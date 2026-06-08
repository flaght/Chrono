from lumina.impulse.fixed import *


def oi039(close, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_log(close)
    weight = roller_sum(chg * openint, weriod, 1, method)
    ont1 = roller_sum(openint, weriod, 1, method)
    factors = -(weight / ont1)

    alpha = roller_mean(factors, window, 1, method)
    return alpha

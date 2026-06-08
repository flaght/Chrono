from lumina.impulse.fixed import *
from lumina.impulse.i009.core.base import calc_umr


# alpha 152
def oi038(value, openint, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    value1 = roller_sum(value, weriod, 1, method)
    ont1 = roller_sum(openint, weriod, 1, method)
    factor = close / (roller_mean(value1, weriod, 1, method) /
                      roller_mean(ont1, weriod, 1, method))

    alpha = roller_mean(factor, window, 1, method)
    return alpha

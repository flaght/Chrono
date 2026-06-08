import pdb
from lumina.impulse.fixed import *


def tf006(high, value, window, weriod, ewm=False):

    method = 'ewm' if ewm else 'rolling'

    core1 = roller_cov(high, value, weriod, weriod, method)

    alpha = roller_mean(core1, window, window, method)
    return alpha

import pdb
from lumina.impulse.fixed import *


def tf004(low, value, window, weriod, ewm=False):

    method = 'ewm' if ewm else 'rolling'
    core1 = roller_cov(roller_max(low, 5, 1, 'rolling'),
                       roller_mean(value, 12, 12, method), 8, 8, method)

    core1 = roller_max(core1, weriod, weriod, 'rolling')

    alpha = roller_mean(core1, window, window, method)

    return alpha

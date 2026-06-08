import pdb
from lumina.impulse.fixed import *


def ha004(value, window, weriod, ewm=False):

    method = 'ewm' if ewm else 'rolling'

    core1 = roller_skew(value, weriod, weriod, method)

    alpha = roller_mean(core1, window, window, method)

    return alpha
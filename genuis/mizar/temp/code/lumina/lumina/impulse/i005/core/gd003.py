import pdb
from lumina.impulse.fixed import *


def gd003(value, volume, close, open, window, weriod, ewm=False):

    method = 'ewm' if ewm else 'rolling'

    core1 = roller_sum(value, weriod, weriod, method) / roller_sum(
        volume, weriod, weriod, method)

    core1 -= open
    core1 /= close.shift(1)

    alpha = roller_mean(core1, window, window, method)

    return alpha

import pdb
from lumina.impulse.fixed import *


def ha005(value, window, weriod, ewm=False):

    method = 'ewm' if ewm else 'rolling'

    core1 = roller_kurt(value, weriod, weriod, 'rolling')

    alpha = roller_mean(core1, window, window, method)

    return alpha
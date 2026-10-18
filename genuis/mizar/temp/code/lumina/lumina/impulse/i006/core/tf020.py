import pdb
from lumina.impulse.fixed import *


def tf020(close, open, low, vwap, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    core1 = np.minimum(roller_max(open, weriod, weriod, 'rolling'),
                       roller_max(close, weriod, weriod, 'rolling'))

    core1 *= low

    core2 = roller_rank(vwap, weriod, weriod, 'rolling').div(low)
    core2 += core1

    alpha = roller_mean(core2, window, window, method)

    return alpha

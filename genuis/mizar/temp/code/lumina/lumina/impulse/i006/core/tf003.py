import pdb
from lumina.impulse.fixed import *


def tf003(close, vwap, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    core1 = vwap.diff(5).sub(np.sqrt(roller_rank(close, weriod, weriod, 'rolling')))

    alpha = roller_mean(core1, window, window, method)

    return alpha

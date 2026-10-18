import pdb
from lumina.impulse.fixed import *


def tf022(volume, low, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    core1 = roller_cov(volume, np.sqrt(close), weriod, weriod, method).div(
        roller_std(volume, weriod, weriod, method)**2)

    core1 = 1 / (1 + np.exp(-core1))

    core2 = roller_cov(core1, low, weriod, weriod,
                       method).div(roller_std(core1, weriod, weriod, method))

    core3 = low - core1 * core2

    alpha = roller_mean(core3, window, window, method)

    return alpha

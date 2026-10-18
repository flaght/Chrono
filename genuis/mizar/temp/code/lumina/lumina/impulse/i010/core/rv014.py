import pdb
from lumina.impulse.fixed import *


def rv014(volume, threshold, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    core1 = volume - roller_mean(volume, weriod, 1, method)
    core2 = roller_quantile(core1, threshold, weriod, 1, 'rolling')
    mask1 = volume < core2

    factor = -core1.mask(mask1, np.nan)

    factor1 = roller_mean(factor, weriod, 1, method)
    factor2 = factor - factor1

    alpha1 = roller_mean(factor1, window, 1, method)
    alpha2 = roller_mean(factor2, window, 1, method)
    return alpha1, alpha2
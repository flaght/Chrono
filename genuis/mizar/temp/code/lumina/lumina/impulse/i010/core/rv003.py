import pdb
import pandas as pd
from lumina.impulse.fixed import *


def rv003(value, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    equal_mean_vwap = roller_mean(value / volume, weriod, 1, method)
    weight_mean_vwap = roller_mean(value, weriod, 1, method) / roller_mean(
        volume, weriod, 1, method)

    factor = equal_mean_vwap / weight_mean_vwap
    factor = np.log(factor)

    factor1 = roller_mean(factor, weriod, 1, method)
    factor2 = factor - factor1

    alpha1 = roller_mean(factor1, window, 1, method)
    alpha2 = roller_mean(factor2, window, 1, method)

    return alpha1, alpha2

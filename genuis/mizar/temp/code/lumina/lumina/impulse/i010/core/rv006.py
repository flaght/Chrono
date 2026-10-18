import pdb
import pandas as pd
from lumina.impulse.fixed import *


def rv006(open, high, low, close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    core1 = volume[((close - open).abs() <= (high - low).abs())
                   & ((close <= open))]

    factor = roller_sum(core1, weriod, 1, method) / roller_sum(
        volume, weriod, 1, method)

    factor1 = roller_mean(factor, weriod, 1, method)
    factor2 = factor - factor1

    alpha1 = roller_mean(factor1, window, 1, method)
    alpha2 = roller_mean(factor2, window, 1, method)

    return alpha1, alpha2

import pdb
import pandas as pd
from lumina.impulse.fixed import *


def rv004(open, high, low, close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    twap = (open + high + low + close) / 4

    mean_twap = roller_mean(twap, weriod, 1, method)
    max_high = roller_max(high, weriod, 1, 'rolling')
    min_low = roller_min(low, weriod, 1, 'rolling')

    factor = (mean_twap - min_low) / (max_high - min_low)

    factor1 = roller_mean(factor, weriod, 1, method)
    factor2 = factor - factor1

    alpha1 = roller_mean(factor1, window, 1, method)
    alpha2 = roller_mean(factor2, window, 1, method)
    return alpha1, alpha2

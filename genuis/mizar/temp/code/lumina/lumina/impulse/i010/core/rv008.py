import pdb
import pandas as pd
from lumina.impulse.fixed import *


def rv008(high, low, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    factor = roller_mean((high - low) / high, weriod, 1, method)

    factor1 = roller_mean(factor, weriod, 1, method)
    factor2 = factor - factor1

    alpha1 = roller_mean(factor1, window, 1, method)
    alpha2 = roller_mean(factor2, window, 1, method)
    return alpha1, alpha2

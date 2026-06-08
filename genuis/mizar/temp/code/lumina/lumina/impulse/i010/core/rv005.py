import pdb
import pandas as pd
from lumina.impulse.fixed import *


def rv005(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_log(close, 1)
    medin_chg = roller_median(chg, weriod, weriod, 'rolling')
    mad = roller_median(1.48 * (chg - medin_chg).abs(), weriod, weriod, 'rolling')
    factor = chg.mask(chg.abs() > 1.5 * mad)

    factor1 = roller_mean(factor, weriod, 1, method)
    factor2 = factor - factor1

    alpha1 = roller_mean(factor1, window, 1, method)
    alpha2 = roller_mean(factor2, window, 1, method)
    return alpha1, alpha2

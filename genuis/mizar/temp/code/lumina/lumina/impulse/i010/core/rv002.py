import pdb
import pandas as pd
from lumina.impulse.fixed import *


def rv002(close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    chg = safe_log(close)

    mean1 = roller_mean(chg, weriod, 1, method)
    std1 = roller_std(chg, weriod, 1, method)

    need_flag = (chg < (mean1 + std1))

    vol_need = volume * need_flag

    factor = roller_std(vol_need, weriod, 1, method) / roller_std(
        volume, weriod, 1, method)


    factor1 = roller_mean(factor, weriod, 1, method)
    factor2 = factor - factor1

    alpha1 = roller_mean(factor1, window, 1, method)
    alpha2 = roller_mean(factor2, window, 1, method)
    return alpha1, alpha2
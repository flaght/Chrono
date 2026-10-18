import pdb
import pandas as pd
from lumina.impulse.fixed import *


### 波动率和伪换手率
def tn006(close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    ## 伪换手率
    close_chg = safe_log(close)
    vol_chg = safe_log(volume)

    activity = roller_mean(close_chg * vol_chg, weriod, 1, method)
    std1 = roller_std(close_chg, weriod, 1, method)

    alpha = activity / std1

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

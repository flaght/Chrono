import pdb
import pandas as pd
from lumina.impulse.fixed import *


### 耀眼收益率
def dv007(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    rets = safe_log(close, 1)
    diff = rets.diff(1)

    dazz_flag = np.where(
        diff > roller_mean(diff, weriod, 1, method) +
        roller_std(diff, weriod, 1, method), 1, 0)

    alpha = roller_mean(dazz_flag * rets, window, 1, method)

    alpha = roller_mean(alpha, window, 1, method)

    return alpha

import pdb
import pandas as pd
from lumina.impulse.fixed import *


### 耀眼波动率
def dv008(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    rets = safe_log(close, 1)
    diff = rets.diff(1)

    dazz_flag = np.where(
        diff > roller_mean(diff, weriod, 1, method) +
        roller_std(diff, weriod, 1, method), 1, 0)

    alpha = roller_std(dazz_flag * rets, window, 1, method)

    alpha = roller_mean(alpha, window, 1, method)

    return alpha

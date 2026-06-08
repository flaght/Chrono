import pdb
import pandas as pd
from lumina.impulse.fixed import *


### 绝对收益与成交量相关性
def dv011(close, value, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    rets = np.abs(safe_log(close, 1))

    alpha = roller_corr(rets, value, weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

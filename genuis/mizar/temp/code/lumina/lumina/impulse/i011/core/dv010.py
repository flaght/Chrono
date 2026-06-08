import pdb
import pandas as pd
from lumina.impulse.fixed import *


### 绝对收益与调整后滞后成交量相关性
def dv010(close, value, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    rets = np.abs(safe_log(close, 1))

    adj_value = (value - roller_mean(value, weriod, 1, method)) / roller_std(
        value, weriod, 1, method)

    alpha = roller_corr(rets, adj_value.shift(1), window, 1, method)

    alpha = roller_mean(alpha, window, 1, method)

    return alpha

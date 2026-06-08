import pdb
import pandas as pd
from lumina.impulse.fixed import *

### 绝对收益与滞后成交量相关性

def dv009(close, value, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    
    rets = safe_log(close, 1)

    alpha = roller_corr(rets, value.shift(1), weriod, 1, method)

    alpha = roller_mean(alpha, window, 1, method)
    return alpha
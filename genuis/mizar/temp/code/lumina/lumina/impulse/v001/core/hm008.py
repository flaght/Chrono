import pdb
import pandas as pd
from lumina.impulse.fixed import *


## 衡量过去N天净流入率的标准差。
## 0值填充
def hm008(buy, sell, hotvalue, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha = (roller_mean(buy, weriod, 1, method) -
             roller_mean(sell, weriod, 1, method)) / (roller_mean(
                 hotvalue, weriod, 1, method) + 1e-6)
    alpha = roller_std(alpha, weriod, 1, method)
    alpha = roller_mean(alpha, window, 1, method)
    alpha = alpha.fillna(0)
    return alpha

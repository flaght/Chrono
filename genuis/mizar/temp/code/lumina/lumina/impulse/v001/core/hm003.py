import pdb
import pandas as pd
from lumina.impulse.fixed import *


## 龙虎榜上买方总金额与卖方总金额的比值，直观反映多空双方的资金实力对比。
## ## 0值填充
def hm003(buy, sell, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha = roller_mean(buy, weriod, 1,
                        method) / (roller_mean(sell, weriod, 1, method) + 1e-6)
    return roller_mean(alpha, window, 1, method)

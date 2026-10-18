import pdb
import pandas as pd
from lumina.impulse.fixed import *


## 衡量龙虎榜上资金对N日总成交的净影响，是判断游资多空意图的核心指标。
## 0值填充
def hm001(buy, sell, value, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha1 = roller_mean(buy, weriod, 1, method) - roller_mean(
        sell, weriod, 1, method)
    alpha = roller_mean(alpha1 / roller_mean(value, weriod, 1, method), window,
                        1, method)
    return alpha

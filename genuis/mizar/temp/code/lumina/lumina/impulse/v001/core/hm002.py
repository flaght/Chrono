import pdb
import pandas as pd
from lumina.impulse.fixed import *


## 衡量龙虎榜席位成交额占当日总成交额的比例，反映大资金对交易的主导程度。
## 0值填充
def hm002(buy, sell, value, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha1 = roller_mean(buy, weriod, 1, method) + roller_mean(
        sell, weriod, 1, method)
    alpha = roller_mean(alpha1 / roller_mean(value, weriod, 1, method), window,
                        1, method)
    return alpha

import pdb
import pandas as pd
from lumina.impulse.fixed import *


## 上榜当日成交量与其过去N日平均成交量的比值，衡量交易活跃度的异常放大程度
## 0值填充
def hm007(hotvalue, value, weriod, window, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha1 = hotvalue / roller_mean(value, weriod, 1, method)
    alpha = roller_mean(alpha1, window, 1, method)

    return alpha

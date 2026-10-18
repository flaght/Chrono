import pdb
import pandas as pd
from lumina.impulse.fixed import *


## 过去N个交易日的累积净买入额，用于平滑单日波动，反映短期资金的真实流向趋势
## 0值填充
def hm005(buy, sell, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha = roller_sum(buy - sell, weriod, 1, method)
    alpha = roller_sum(alpha, window, 1, method)
    return alpha

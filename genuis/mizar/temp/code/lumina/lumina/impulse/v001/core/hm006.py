import pdb
import pandas as pd
from lumina.impulse.fixed import *


### 净流入变化率 正值表示买入力度在加速增强，可能是行情启动信号；负值则表示动能衰减，需要警惕
## 0值填充
def hm006(buy, sell, hotvalue, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    alpha1 = (roller_mean(buy, weriod, 1, method) -
              roller_mean(sell, weriod, 1, method) /
              roller_mean(hotvalue, weriod, 1, method))

    alpha = roller_mean(alpha1, window, 1, method)
    return alpha
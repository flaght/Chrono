import pdb
import pandas as pd
from lumina.impulse.fixed import *


## 鳄鱼线计算
## 1. 计算三条均值
## 2. 上唇线 < 牙齿线 < 下唇线 空
## 3. 上唇线 > 牙齿线 > 下唇线 多
def tn003(close, long, medium, short, window, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    jaw = roller_mean(close, long, 1, method)
    teeth = roller_mean(close, medium, 1, method)
    lips = roller_mean(close, short, 1, method)

    alpha = (jaw - teeth) - (teeth - lips)

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

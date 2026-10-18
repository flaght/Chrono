import pdb
import pandas as pd
from lumina.impulse.fixed import *


##日内动量 上下边界商
def tn004(close, open, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    ## 计算sigma
    ### 开盘价和收盘价的距离
    distance = (close / open) - 1
    sigma = roller_mean(distance, weriod, 1, method)

    up_bound = np.maximum(open, close.shift(1)) * (1 + sigma)
    low_bound = np.minimum(open, close.shift(1)) * (1 - sigma)

    alpha = up_bound / low_bound

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

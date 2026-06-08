import pdb
import pandas as pd
from lumina.impulse.fixed import *


## 价量共振指标
def tn001(close, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    ### 成交量移动平均线
    ama = roller_mean(volume, weriod, 1, method)
    ### 收盘价移动平均线
    bma = roller_mean(close, weriod, 1, method)
    ### 价能
    bma_chg = bma / bma.shift(window)
    ### 量能
    ama_chg = ama / ama.shift(window)
    
    ## 价能 * 量能
    alpha = bma_chg * ama_chg

    alpha = roller_mean(alpha, window, 1, method)
    return alpha

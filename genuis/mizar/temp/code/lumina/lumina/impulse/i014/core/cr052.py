"""
cr052：持仓量(openint)与长期均值偏离度的tanh非线性压缩因子，衡量持仓均值回归强度。
计算方式：持仓量与N期均值之差标准化后，经过tanh变换，滑动平均。
本因子为cr042的持仓量(openint)版本。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *

def cr052(openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    mean_long = roller_mean(openint, weriod, 1, method)
    std_long = roller_std(openint, weriod, 1, method)
    zscore = (openint - mean_long) / (std_long + 1e-8)
    tanh_val = np.tanh(zscore)
    factor = roller_mean(tanh_val, window, 1, method)
    return factor 
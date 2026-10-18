"""
cr042：收盘价与长期均值偏离度的tanh非线性压缩因子，衡量均值回归强度。
计算方式：收盘价与N期均值之差标准化后，经过tanh变换，滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr042(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    mean_long = roller_mean(close, weriod, 1, method)
    std_long = roller_std(close, weriod, 1, method)
    zscore = (close - mean_long) / (std_long + 1e-8)
    tanh_val = np.tanh(zscore)
    factor = roller_mean(tanh_val, window, 1, method)
    return factor

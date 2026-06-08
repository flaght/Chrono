# -*- encoding:utf-8 -*-
"""
    计算线rsi模块
"""
import numpy as np
import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean

def calc_rsi(kl_pd, xd, scalar=None, ewm=True):

    negative = kl_pd['close'].diff()
    positive = negative.copy()

    #negative[negative > 0] = 0
    #positive[positive < 0] = 0
    negative = pd.Series(np.where(negative > 0, 0, negative))
    positive = pd.Series(np.where(positive < 0, 0, positive))

    if ewm:
        negative_avg = pd_ewm_mean(negative, span=xd, min_periods=xd)
        positive_avg = pd_ewm_mean(positive, span=xd, min_periods=xd)
    else:
        negative_avg = pd_rolling_mean(negative, window=xd, min_periods=xd)
        positive_avg = pd_rolling_mean(positive, window=xd, min_periods=xd)

    rsi = scalar * negative_avg / (positive_avg + negative_avg.abs())

    line = Line(rsi, 'rsi')
    return line

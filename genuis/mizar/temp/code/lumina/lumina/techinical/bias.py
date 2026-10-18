# -*- encoding:utf-8 -*-
"""
    计算线bias模块
"""
import pandas as pd
from ultron.ump.core.helper import pd_rolling_mean,pd_ewm_mean
from ultron.ump.technical.line import Line
import pdb

def calc_bias(kl_pd, xd, ewm=True):
    close = kl_pd['close']
    if ewm:
        ma = pd_ewm_mean(close, span=xd, min_periods=xd)
    else:
        ma = pd_rolling_mean(close, window=xd, min_periods=xd)
    bias = (close - ma) / ma * 100
    bias = pd.Series(bias).fillna(method='bfill')
    line = Line(bias, 'bias')
    return line
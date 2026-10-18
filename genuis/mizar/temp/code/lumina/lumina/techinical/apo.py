# -*- encoding:utf-8 -*-
"""
    计算线ao模块
"""
import numpy as np
import pandas as pd
from ultron.ump.core.helper import pd_rolling_mean,pd_ewm_mean
from ultron.ump.technical.line import Line

def calc_apo(kl_pd, fast, slow,  ewm=True):
    close = kl_pd['close']
    
    if ewm:
        apo = pd_ewm_mean(close, span=fast, min_periods=fast) - pd_ewm_mean(close, span=slow, min_periods=slow)
    else:
        apo = pd_rolling_mean(close, window=fast, min_periods=fast) - pd_rolling_mean(close, window=slow, min_periods=slow)

    apo = pd.Series(apo).fillna(method='bfill')
    line = Line(apo, 'apo')
    return line
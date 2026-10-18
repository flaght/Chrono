# -*- encoding:utf-8 -*-
"""
    计算线ao模块
"""
import numpy as np
import pandas as pd
from ultron.ump.core.helper import pd_rolling_mean,pd_ewm_mean
from ultron.ump.technical.line import Line

def calc_ao(kl_pd, fast, slow, ewm=True):
    high = kl_pd['high']
    low = kl_pd['low']

    median_price = (high + low) / 2
    
    if ewm:
        ao = pd_ewm_mean(median_price, span=fast, min_periods=fast) - pd_ewm_mean(median_price, span=slow, min_periods=slow)
    else:
        ao = pd_rolling_mean(median_price, window=fast, min_periods=fast) - pd_rolling_mean(median_price, window=slow, min_periods=slow)
    ao = pd.Series(ao).fillna(method='bfill')
    line = Line(ao, 'ao')
    return line
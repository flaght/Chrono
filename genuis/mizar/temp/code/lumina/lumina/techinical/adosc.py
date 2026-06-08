# -*- encoding:utf-8 -*-
"""
    计算adosc模块
"""

import numpy as np
import pandas as pd
from ultron.ump.core.helper import pd_rolling_mean,pd_ewm_mean
from ultron.ump.technical.line import Line

def calc_adosc(kl_pd, fast, slow, ewm=True):
    fast = int(fast) if fast and fast > 0 else 3
    slow = int(slow) if slow and slow > 0 else 10
    
    high = kl_pd['high']
    low = kl_pd['low']
    close = kl_pd['close']
    volume = kl_pd['volume']
    ad = ((close - low) - (high - close)) / (high - low) * volume

    if ewm:
        adosc = pd_ewm_mean(ad, span=fast, min_periods=fast) - pd_ewm_mean(ad, span=slow, min_periods=slow)
    else:
        adosc = pd_rolling_mean(ad, window=fast, min_periods=fast) - pd_rolling_mean(ad, window=slow, min_periods=slow)

    adosc = pd.Series(adosc).fillna(method='bfill')
    line = Line(adosc, 'adosc')
    return line

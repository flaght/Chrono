# -*- encoding:utf-8 -*-
"""
    计算线MASSI模块
"""
import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean
import pdb

def calc_massi(kl_pd, fast, slow, ewm=True):
    
    fast = int(fast) if fast and fast > 0 else 9
    slow = int(slow) if slow and slow > 0 else 25

    high_low_range = (kl_pd['high'] - kl_pd['low']).fillna(0)

    if ewm:
        hl_ema1 = pd_ewm_mean(high_low_range, span=fast, min_periods=fast)
        hl_ema2 = pd_ewm_mean(high_low_range, span=slow, min_periods=slow)
    else:
        hl_ema1 = pd_rolling_mean(high_low_range, window=fast, min_periods=fast)
        hl_ema2 = pd_rolling_mean(high_low_range, window=slow, min_periods=slow)

    hi_ratio = hl_ema1 / hl_ema2
    if ewm:
        massi = pd_ewm_mean(hi_ratio, span=slow, min_periods=slow)
    else:
        massi = pd_rolling_mean(hi_ratio, window=slow, min_periods=slow)

    massi = pd.Series(massi).fillna(method='bfill')
    line = Line(massi, 'massi')

    return line
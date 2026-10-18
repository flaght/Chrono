# -*- encoding:utf-8 -*-
"""
    计算线KVO模块
"""
import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean
import pdb


def calc_kvo(kl_pd, fast, slow, drift, ewm=False):
    fast = int(fast) if fast and fast > 0 else 34
    slow = int(slow) if slow and slow > 0 else 55
    drift = int(drift) if isinstance(drift, int) and drift != 0 else 1
    
    high = kl_pd['high']
    low = kl_pd['low']
    close = kl_pd['close']
    volume = kl_pd['volume']

    hlc3 = (high + low + close) / 3.0
    sign = hlc3.diff(1)
    sign[sign > 0] = 1
    sign[sign < 0] = -1
    sign.iloc[0] = 1

    signal_vol = sign * volume

    sv = signal_vol.loc[
        signal_vol.first_valid_index():,
    ]

    if ewm:
        kvo = pd_ewm_mean(signal_vol, span=fast,
                          min_periods=fast) - pd_ewm_mean(
                              signal_vol, span=slow, min_periods=slow)
    else:
        kvo = pd_rolling_mean(signal_vol, window=fast,
                              min_periods=fast) - pd_rolling_mean(
                                  signal_vol, window=slow, min_periods=slow)
        
    kvo = pd.Series(kvo).fillna(method='bfill')
    line = Line(kvo, 'kvo')
    return line

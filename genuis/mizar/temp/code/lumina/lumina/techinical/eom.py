# -*- encoding:utf-8 -*-
"""
    计算线EOM模块
"""

import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean
import pdb

def calc_eom(kl_pd, xd, drift, divisor, ewm=False):

    xd = int(xd) if xd and xd > 0 else 14
    divisor = divisor if divisor and divisor > 0 else 100000000
    drift = int(drift) if isinstance(drift, int) and drift != 0 else 1


    high = kl_pd['high']
    low = kl_pd['low']
    volume = kl_pd['volume']

    distance = 0.5 * (high + low)
    distance -= 0.5 * (high.shift(drift) + low.shift(drift))
    box_ratio = volume / divisor / (high - low)
    eom = distance * box_ratio
    if ewm:
        eom = pd_ewm_mean(eom, span=xd, min_periods=xd)
    else:
        eom = pd_rolling_mean(eom, window=xd, min_periods=xd)

    eom = pd.Series(eom).fillna(method='bfill')
    line = Line(eom, 'eom')
    return line
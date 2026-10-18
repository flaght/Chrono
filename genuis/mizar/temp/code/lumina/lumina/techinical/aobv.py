# -*- encoding:utf-8 -*-
"""
    计算aobv模块
"""

import numpy as np
import pandas as pd
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean
from ultron.ump.technical.line import Line


def calc_aobv(kl_pd,
              fast,
              slow,
              ewm=True):
    fast = int(fast) if fast and fast > 0 else 4
    slow = int(slow) if slow and slow > 0 else 12

    close = kl_pd['close']

    sign = close.diff(1)
    sign[sign > 0] = 1
    sign[sign < 0] = -1
    sign.iloc[0] = 1

    signed_volume = sign * kl_pd['volume']
    obv = signed_volume.cumsum()

    if ewm:
        maf = pd_ewm_mean(obv, span=fast, min_periods=fast)
        mas = pd_ewm_mean(obv, span=slow, min_periods=slow)
    else:
        maf = pd_rolling_mean(obv, window=fast, min_periods=fast)
        mas = pd_rolling_mean(obv, window=slow, min_periods=slow)

    aobv = maf - mas
    aobv = pd.Series(aobv).fillna(method='bfill')
    line = Line(aobv, 'aobv')
    return line






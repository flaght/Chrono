# -*- encoding:utf-8 -*-
"""
    计算cmf模块
"""

import numpy as np
import pandas as pd
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean
from ultron.ump.technical.line import Line

def calc_cmf(kl_pd, xd, ewm=True):
    xd = int(xd) if xd and xd > 0 else 20

    ad = 2 * kl_pd['close'] - (kl_pd['high'] + kl_pd['low'])

    ad *= kl_pd['volume'] / (kl_pd['high'] - kl_pd['low']).fillna(0)

    if ewm:
        cmf = pd_ewm_mean(ad, span=xd, min_periods=xd)
    else:
        cmf = pd_rolling_mean(ad, window=xd, min_periods=xd)
    cmf = pd.Series(cmf).fillna(method='bfill')
    line = Line(cmf, 'cmf')
    return line
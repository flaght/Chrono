# -*- encoding:utf-8 -*-
"""
    计算线psl模块
"""

import numpy as np
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean



def calc_psl(kl_pd, xd, ewm=True):
    diff = np.sign(kl_pd['close'] - kl_pd['open'])
    diff.fillna(0, inplace=True)
    diff[diff <= 0] = 0

    if ewm:
        psl = pd_ewm_mean(diff, span=xd, min_periods=xd)
    else:
        psl = pd_rolling_mean(diff, window=xd, min_periods=xd)

    line = Line(psl, 'psl')
    return line
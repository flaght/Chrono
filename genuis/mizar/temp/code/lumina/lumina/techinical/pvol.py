# -*- encoding:utf-8 -*-
"""
    计算线PVOL模块
"""
import pandas as pd
import numpy as np
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean
import pdb


def calc_pvol(kl_pd, offset=0, **kwargs):

    close = kl_pd['close']
    volume = kl_pd['volume']
    pvol = close * volume

    sign = close.diff(1)
    sign = pd.Series(np.where(sign > 0, 1, sign))
    sign = pd.Series(np.where(sign < 0, -1, sign))

    #sign[sign > 0] = 1
    #sign[sign < 0] = -1
    #sign.iloc[0] = 1

    pvol = sign.values * pvol.values

    pvol = pd.Series(pvol).fillna(method='bfill')
    line = Line(pvol, 'pvol')
    return line

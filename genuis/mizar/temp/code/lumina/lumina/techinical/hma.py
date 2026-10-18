# -*- encoding:utf-8 -*-
"""
    计算线HMA模块
"""
import numpy as np
import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean
import pdb


def calc_hma(kl_pd, length, ewm=True):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))

    if ewm:
        wmaf = pd_ewm_mean(kl_pd['close'],
                           span=half_length,
                           min_periods=half_length)
        wmas = pd_ewm_mean(kl_pd['close'], span=length, min_periods=length)
        hma = pd_ewm_mean(2 * wmaf - wmas,
                          span=sqrt_length,
                          min_periods=sqrt_length)
    else:
        wmaf = pd_rolling_mean(kl_pd['close'],
                               window=half_length,
                               min_periods=half_length)
        wmas = pd_rolling_mean(kl_pd['close'],
                               window=length,
                               min_periods=length)
        hma = pd_rolling_mean(2 * wmaf - wmas,
                              window=sqrt_length,
                              min_periods=sqrt_length)

    hma = pd.Series(hma).fillna(method='bfill')
    line = Line(hma, 'hma')

    return line

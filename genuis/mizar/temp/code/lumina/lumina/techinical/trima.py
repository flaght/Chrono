# -*- encoding:utf-8 -*-
"""
    计算线TRIMA模块
"""
import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean
import pdb


def calc_trima(kl_pd, length, ewm=True):
    close = kl_pd['close']
    half_length = round(0.5 * (length + 1))
    if ewm:
        trima = pd_ewm_mean(pd_ewm_mean(close,
                                        span=half_length,
                                        min_periods=half_length),
                            span=half_length,
                            min_periods=half_length)
    else:
        trima = pd_rolling_mean(pd_rolling_mean(close,
                                                window=half_length,
                                                min_periods=half_length),
                                window=half_length,
                                min_periods=half_length)

    trima = pd.Series(trima).fillna(method='bfill')
    line = Line(trima, 'trima')
    return line

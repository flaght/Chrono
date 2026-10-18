# -*- encoding:utf-8 -*-
"""
    计算线TEMA模块
"""
import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean
import pdb


def calc_tema(kl_pd, length, ewm=True):
    close = kl_pd['close']

    if ewm:
        ema = pd_ewm_mean(close, span=length, min_periods=length)
        ema_ema = pd_ewm_mean(ema, span=length, min_periods=length)
        ema_ema_ema = pd_ewm_mean(ema_ema, span=length, min_periods=length)
    else:
        ema = pd_rolling_mean(close, window=length, min_periods=length)
        ema_ema = pd_rolling_mean(ema, window=length, min_periods=length)
        ema_ema_ema = pd_rolling_mean(ema_ema,
                                      window=length,
                                      min_periods=length)

    tema = 3 * ema - 3 * ema_ema + ema_ema_ema
    tema = pd.Series(tema).fillna(method='bfill')
    line = Line(tema, 'tema')
    return line

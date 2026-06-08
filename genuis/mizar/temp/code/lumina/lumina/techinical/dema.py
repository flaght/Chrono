# -*- encoding:utf-8 -*-
"""
    计算线DEMA模块
"""
import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean
import pdb


def calc_dema(kl_pd, xd, ewm=True):

    xd = int(xd) if xd and xd > 0 else 10

    close = kl_pd['close']

    if ewm:
        ema = pd_ewm_mean(close, span=xd, min_periods=xd)
        ema_ema = pd_ewm_mean(ema, span=xd, min_periods=xd)
    else:
        ema = pd_rolling_mean(close, window=xd, min_periods=xd)
        ema_ema = pd_rolling_mean(ema, window=xd, min_periods=xd)

    dema = 2 * ema - ema_ema
    dema = pd.Series(dema).fillna(method='bfill')
    line = Line(dema, 'dema')
    return line

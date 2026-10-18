# -*- encoding:utf-8 -*-
"""
    计算线KC模块
"""
import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean
import pdb


def calc_kc(kl_pd, xd, scalar, ewm=True):

    xd = int(xd) if xd and xd > 0 else 20
    scalar = float(scalar) if scalar and scalar > 0 else 2

    range_ = (kl_pd['high'] - kl_pd['low']).fillna(0)

    if ewm:
        band = pd_ewm_mean(range_, span=xd, min_periods=xd)
        basis = pd_ewm_mean(kl_pd['close'], span=xd, min_periods=xd)
    else:
        band = pd_rolling_mean(range_, window=xd, min_periods=xd)
        basis = pd_rolling_mean(kl_pd['close'], window=xd, min_periods=xd)

    upper = basis + scalar * band
    lower = basis - scalar * band

    upper = pd.Series(upper).fillna(method='bfill')
    lower = pd.Series(lower).fillna(method='bfill')

    return Line(upper, 'upper'), Line(lower, 'lower')

# -*- encoding:utf-8 -*-
"""
    计算线THERMO模块
"""
import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean
import pdb


def calc_thermo(kl_pd, xd, drift, ewm=True):

    xd = int(xd) if xd and xd > 0 else 20
    drift = int(drift) if isinstance(drift, int) and drift != 0 else 1

    thermo_l = (kl_pd['low'].shift(drift) - kl_pd['low']).abs()
    thermo_h = (kl_pd['high'].shift(drift) - kl_pd['high']).abs()

    thermo = thermo_l
    thermo = thermo.where(thermo_l > thermo_h, thermo_h)

    if ewm:
        thermo = pd_ewm_mean(thermo, span=xd, min_periods=xd)
    else:
        thermo = pd_rolling_mean(thermo, window=xd, min_periods=xd)

    thermo = pd.Series(thermo).fillna(method='bfill')
    line = Line(thermo, 'thermo')
    return line

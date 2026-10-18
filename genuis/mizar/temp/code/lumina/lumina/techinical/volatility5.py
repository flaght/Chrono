# -*- encoding:utf-8 -*-
"""
    计算线volatility5模块
"""
import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean
import pdb


def calc_volatility5(kl_pd, xd, drift, ewm=True):
    xd = int(xd) if xd and xd > 0 else 10
    drift = int(drift) if drift and drift > 0 else 5

    high = kl_pd['high']
    low = kl_pd['low']

    if ewm:
        volatility = pd_ewm_mean(high - low, span=xd, min_periods=drift)
    else:
        volatility = pd_rolling_mean(high - low, window=xd, min_periods=drift)

    volatility = pd.Series(volatility).fillna(method='bfill')
    line = Line(volatility, 'volatility5')
    return line

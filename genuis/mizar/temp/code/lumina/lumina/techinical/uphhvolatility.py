# -*- encoding:utf-8 -*-
"""
    计算线upllvolatility模块
"""

from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean


def calc_uphhvolatility(kl_pd, xd, drift, ewm=True):
    if ewm:
        uphhvolatility = pd_ewm_mean((kl_pd['high'] - kl_pd['high'].shift(drift)) / kl_pd['high'].shift(drift), span=xd, min_periods=xd)
    else:
        uphhvolatility = pd_rolling_mean((kl_pd['high'] - kl_pd['high'].shift(drift)) / kl_pd['high'].shift(drift), window=xd, min_periods=xd)
    line = Line(uphhvolatility, 'uphhvolatility')
    return line
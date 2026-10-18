# -*- encoding:utf-8 -*-
"""
    计算线upchvolatility模块
"""

from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean


def calc_upchvolatility(kl_pd, xd, drift, ewm=True):
    if ewm:
        upchvolatility = pd_ewm_mean((kl_pd['high'] - kl_pd['close'].shift(drift)) / kl_pd['close'].shift(drift), span=xd, min_periods=xd)
    else:
        upchvolatility = pd_rolling_mean((kl_pd['high'] - kl_pd['close'].shift(drift)) / kl_pd['close'].shift(drift), window=xd, min_periods=xd)
    line = Line(upchvolatility, 'upchvolatility')
    return line
# -*- encoding:utf-8 -*-
"""
    计算线upintraday模块
"""

from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean


def calc_upintraday(kl_pd, xd, drift, ewm=True):
    if ewm:
        upintraday = pd_ewm_mean((kl_pd['high'] - kl_pd['open'].shift(drift)) / kl_pd['open'].shift(drift), span=xd, min_periods=xd)
    else:
        upintraday = pd_rolling_mean((kl_pd['high'] - kl_pd['open'].shift(drift)) / kl_pd['open'].shift(drift), window=xd, min_periods=xd)
    line = Line(upintraday, 'upintraday')
    return line
# -*- encoding:utf-8 -*-
"""
    计算线dnintraday模块
"""

from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean


def calc_dnintraday(kl_pd, xd, drift, ewm=True):
    if ewm:
        dnintraday = pd_ewm_mean((kl_pd['low'] - kl_pd['open'].shift(drift)) / kl_pd['open'].shift(drift), span=xd, min_periods=xd)
    else:
        dnintraday = pd_rolling_mean((kl_pd['low'] - kl_pd['open'].shift(drift)) / kl_pd['open'].shift(drift), window=xd, min_periods=xd)
    line = Line(dnintraday, 'dnintraday')
    return line
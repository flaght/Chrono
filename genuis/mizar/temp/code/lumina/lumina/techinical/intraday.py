# -*- encoding:utf-8 -*-
"""
    计算线interaday模块
"""

from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean


def calc_intraday(kl_pd, xd, drift, ewm=True):
    if ewm:
        intraday = pd_ewm_mean((kl_pd['close'] - kl_pd['open'].shift(drift)) / kl_pd['open'].shift(drift), span=xd, min_periods=xd)
    else:
        intraday = pd_rolling_mean((kl_pd['close'] - kl_pd['open'].shift(drift)) / kl_pd['open'].shift(drift), window=xd, min_periods=xd)
    line = Line(intraday, 'intraday')
    return line
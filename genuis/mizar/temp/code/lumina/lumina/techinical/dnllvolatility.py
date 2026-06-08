# -*- encoding:utf-8 -*-
"""
    计算线dnllvolatility模块
"""
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean


def calc_dnllvolatility(kl_pd, xd, drift, ewm=True):
    if ewm:
        dnllvolatility = pd_ewm_mean((kl_pd['close'] - kl_pd['low'].shift(drift)) / kl_pd['low'].shift(drift), span=xd, min_periods=xd)
    else:
        dnllvolatility = pd_rolling_mean((kl_pd['close'] - kl_pd['low'].shift(drift)) / kl_pd['low'].shift(drift), window=xd, min_periods=xd)
    line = Line(dnllvolatility, 'dnllvolatility')
    return line
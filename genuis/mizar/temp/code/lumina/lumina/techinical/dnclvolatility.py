# -*- encoding:utf-8 -*-
"""
    计算线dnclvolatility模块
"""

from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean


def calc_dnclvolatility(kl_pd, xd, drift, ewm=True):
    if ewm:
        dnclvolatility = pd_ewm_mean((kl_pd['close'] - kl_pd['close'].shift(drift)) / kl_pd['close'].shift(drift), span=xd, min_periods=xd)
    else:
        dnclvolatility = pd_rolling_mean((kl_pd['close'] - kl_pd['close'].shift(drift)) / kl_pd['close'].shift(drift), window=xd, min_periods=xd)
    line = Line(dnclvolatility, 'dnclvolatility')
    return line
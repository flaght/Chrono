# -*- encoding:utf-8 -*-
"""
    计算线roc模块
"""

from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean

def calc_roc(kl_pd, xd, scalar, drift, ewm=True):
    roc = scalar * (kl_pd['close'] - kl_pd['close'].shift(drift)) / kl_pd['close'].shift(drift)
    if ewm:
        roc = pd_ewm_mean(roc, span=xd, min_periods=xd)
    else:
        roc = pd_rolling_mean(roc, window=xd, min_periods=xd)

    line = Line(roc, 'roc')
    return line
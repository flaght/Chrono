# -*- encoding:utf-8 -*-
"""
    计算线tsix模块
"""

from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean


def calc_tsix(kl_pd, xd, scalar, drift, ewm=True):
    if ewm:
        ema1 = pd_ewm_mean(kl_pd['close'], span=xd, min_periods=xd)
        ema2 = pd_ewm_mean(ema1, span=xd, min_periods=xd)
        ema3 = pd_ewm_mean(ema2, span=xd, min_periods=xd)
    else:
        ema1 = pd_rolling_mean(kl_pd['close'], window=xd, min_periods=xd)
        ema2 = pd_rolling_mean(ema1, window=xd, min_periods=xd)
        ema3 = pd_rolling_mean(ema2, window=xd, min_periods=xd)
    
    trix = scalar * ema3.pct_change(drift)

    line = Line(trix, 'tsix')

    return line

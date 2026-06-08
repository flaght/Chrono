# -*- encoding:utf-8 -*-
"""
    计算线T3模块
"""
import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean
import pdb

def calc_t3(kl_pd, xd, a=None, ewm=True):
    xd = int(xd) if xd and xd > 0 else 10
    a = float(a) if a and a > 0 and a < 1 else 0.7

    close = kl_pd['close']
    c1 = -a * a**2
    c2 = 3 * a**2 + 3 * a**3
    c3 = -6 * a**2 - 3 * a - 3 * a**3
    c4 = a**3 + 3 * a**2 + 3 * a + 1
    if ewm:
        e1 = pd_ewm_mean(close, span=xd, min_periods=1)
        e2 = pd_ewm_mean(e1, span=xd, min_periods=1)
        e3 = pd_ewm_mean(e2, span=xd, min_periods=1)
        e4 = pd_ewm_mean(e3, span=xd, min_periods=1)
        e5 = pd_ewm_mean(e4, span=xd, min_periods=1)
        e6 = pd_ewm_mean(e5, span=xd, min_periods=1)
    else:
        e1 = pd_rolling_mean(close, window=xd, min_periods=1)
        e2 = pd_rolling_mean(e1, window=xd, min_periods=1)
        e3 = pd_rolling_mean(e2, window=xd, min_periods=1)
        e4 = pd_rolling_mean(e3, window=xd, min_periods=1)
        e5 = pd_rolling_mean(e4, window=xd, min_periods=1)
        e6 = pd_rolling_mean(e5, window=xd, min_periods=1)

    t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    t3 = pd.Series(t3).fillna(method='bfill')
    line = Line(t3, 't3')
    return line
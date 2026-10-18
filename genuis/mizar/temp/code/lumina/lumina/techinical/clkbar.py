# -*- encoding:utf-8 -*-
"""
    计算线clkbar模块
"""
import pandas 
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean

def calc_clkbar(kl_pd, xd, ewm=True):
    if ewm:
        clkbar = pd_ewm_mean((kl_pd['close'] - kl_pd['low']) / kl_pd['low'], span=xd, min_periods=xd)
    else:
        clkbar = pd_rolling_mean((kl_pd['close'] - kl_pd['low']) / kl_pd['low'], window=xd, min_periods=xd)
    line = Line(clkbar, 'clkbar')
    return line
# -*- encoding:utf-8 -*-
"""
    计算线chkbar模块
"""
import pdb
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean

def calc_chkbar(kl_pd, xd, ewm=True):
    if ewm:
        chkbar = pd_ewm_mean((kl_pd['close'] - kl_pd['high']) / kl_pd['high'], span=xd, min_periods=xd)
    else:
        chkbar = pd_rolling_mean((kl_pd['close'] - kl_pd['high']) / kl_pd['high'], window=xd, min_periods=xd)
    line = Line(chkbar, 'chkbar')
    return line
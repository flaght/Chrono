# -*- encoding:utf-8 -*-
"""
    计算线er模块
"""

from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean

def calc_er(kl_pd, xd, drift, ewm=True):
    abs_diff = kl_pd['close'].diff(xd).abs()
    abs_volatility = kl_pd['close'].diff(drift).abs()

    if ewm:
        er = abs_diff / pd_ewm_mean(abs_volatility, span=xd, min_periods=xd)
    else:
        er = abs_diff / pd_rolling_mean(abs_volatility, window=xd, min_periods=xd)
    
    line = Line(er, 'er')
    return line
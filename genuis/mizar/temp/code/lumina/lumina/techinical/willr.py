# -*- encoding:utf-8 -*-
"""
    计算线willr模块
"""

from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_min, pd_rolling_max


def calc_willr(kl_pd, xd):
    lowest = pd_rolling_min(kl_pd['low'], window=xd, min_periods=xd)
    highest = pd_rolling_max(kl_pd['high'], window=xd, min_periods=xd)
    
    willr = 100 * ((kl_pd['close'] - lowest) / (highest - lowest) - 1)

    line = Line(willr, 'willr')
    return line
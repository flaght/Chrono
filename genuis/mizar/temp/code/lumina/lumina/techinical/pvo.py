# -*- encoding:utf-8 -*-
"""
    计算线pvo模块
"""

from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean


def calc_pvo(kl_pd, fast, slow, scalar, ewm=True):
    if ewm:
        fast_ma = pd_ewm_mean(kl_pd['volume'], span=fast, min_periods=fast)
        slow_ma = pd_ewm_mean(kl_pd['volume'], span=slow, min_periods=slow)
    else:
        fast_ma = pd_rolling_mean(kl_pd['volume'], window=fast, min_periods=fast)
        slow_ma = pd_rolling_mean(kl_pd['volume'], window=slow, min_periods=slow)
    pvo = scalar * (fast_ma - slow_ma) / slow_ma
    line = Line(pvo, 'pvo')
    return line
# -*- encoding:utf-8 -*-
"""
    计算线brar模块
"""
import pdb
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_sum, pd_ewm_sum


def calc_brar(kl_pd, xd, scalar, drift, ewm=True):
    if ewm:
        ar = pd_ewm_sum((kl_pd['high'] - kl_pd['open']), 
                        span=xd, min_periods=xd) / pd_ewm_sum(
                            (kl_pd['open'] - kl_pd['low']), span=xd, 
                            min_periods=xd) * scalar
        
        br = pd_ewm_sum((kl_pd['high'] - kl_pd['close'].shift(drift)), 
                        span=xd, min_periods=xd) / pd_ewm_sum(
                            (kl_pd['close'].shift(drift) - kl_pd['low']), span=xd, 
                            min_periods=xd) * scalar
    else:
        ar = pd_rolling_sum((kl_pd['high'] - kl_pd['open']), 
                            window=xd, min_periods=xd) / pd_rolling_sum(
                                (kl_pd['open'] - kl_pd['low']), window=xd, 
                                min_periods=xd) * scalar
        br = pd_rolling_sum((kl_pd['high'] - kl_pd['close'].shift(drift)), 
                            window=xd, min_periods=xd) / pd_rolling_sum(
                                (kl_pd['close'].shift(drift) - kl_pd['low']), window=xd, 
                                min_periods=xd) * scalar
    brar = ar - br

    line = Line(brar, 'brar')
    return line
# -*- encoding:utf-8 -*-
"""
    计算线pgo模块
"""
from ultron.ump.indicator.atr import calc_atr
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean


def calc_pgo(kl_pd, xd, ewm=True):
    if ewm:
        pgo = kl_pd['close'] - pd_ewm_mean(kl_pd['close'], span=xd, min_periods=xd)
    else:
        pgo = kl_pd['close'] - pd_rolling_mean(kl_pd['close'], window=xd, min_periods=xd)

    atr_line = calc_atr(kl_pd['high'], 
                        kl_pd['low'], 
                        kl_pd['close'], xd)
    pgo = pgo / atr_line

    line = Line(pgo, 'pgo')
    return line
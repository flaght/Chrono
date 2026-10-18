# -*- encoding:utf-8 -*-
"""
    计算线stochrsi模块
"""
import pdb
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_rolling_max, pd_rolling_min, pd_ewm_mean


def calc_stochrsi(kl_pd, xd, rsi_xd, fast_xd, scalar, ewm=True):
    negative = kl_pd['close'].diff()
    positive = negative.copy()

    negative[negative > 0] = 0
    positive[positive < 0] = 0

    if ewm:
        negative_avg = pd_ewm_mean(negative, span=rsi_xd, min_periods=rsi_xd)
        positive_avg = pd_ewm_mean(positive, span=rsi_xd, min_periods=rsi_xd)
    else:
        negative_avg = pd_rolling_mean(negative, window=rsi_xd, min_periods=rsi_xd)
        positive_avg = pd_rolling_mean(positive, window=rsi_xd, min_periods=rsi_xd)

    rsi = scalar * negative_avg / (positive_avg + negative_avg.abs())

    lowest_rsi = pd_rolling_min(rsi, window=xd, min_periods=xd)
    highest_rsi = pd_rolling_max(rsi, window=xd, min_periods=xd)

    stoch_rsi = pd_ewm_mean(highest_rsi - lowest_rsi, span=fast_xd, min_periods=fast_xd)
    line = Line(stoch_rsi, 'stochrsi')
    return line
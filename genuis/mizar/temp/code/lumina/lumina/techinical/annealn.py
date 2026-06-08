# -*- encoding:utf-8 -*-
"""
    计算线annealn模块
"""
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean
import pdb

def calc_annealn(kl_pd, xd=14, ewm=False):
    high_low_range = (kl_pd['high'] - kl_pd['low']).abs()
    high_close_range = (kl_pd['high'] - kl_pd['close'].shift(1)).abs()
    low_close_range = (kl_pd['low'] - kl_pd['close'].shift(1)).abs()

    cond1 = high_close_range > high_low_range
    cond2 = high_close_range > low_close_range

    high_close_range[cond1] = high_low_range[cond1]
    low_close_range[cond2] = high_close_range[cond2]

    tr = low_close_range
    if ewm:
        atr = pd_ewm_mean(tr, span=xd, min_periods=xd, adjust=True)
    else:
        atr = pd_rolling_mean(tr, window=xd, min_periods=xd)
    ret = kl_pd['close'] - kl_pd['close'].shift(xd) + 0.00001
    atr_adj = 2 * ret / (atr + atr.shift(xd))
    line = Line(atr_adj, 'annealn')
    return line
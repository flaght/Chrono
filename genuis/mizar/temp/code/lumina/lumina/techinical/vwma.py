# -*- encoding:utf-8 -*-
"""
    计算线VMA模块
"""
import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean
import pdb


def calc_vwma(kl_pd, xd, ewm=True):
    xd = int(xd) if xd and xd > 0 else 10

    close = kl_pd['close']
    volume = kl_pd['volume']

    if ewm:
        vma = pd_ewm_mean(close * volume, span=xd, min_periods=xd)
        vma /= pd_ewm_mean(volume, span=xd, min_periods=xd)
    else:
        vma = pd_rolling_mean(close * volume, window=xd, min_periods=xd)
        vma /= pd_rolling_mean(volume, window=xd, min_periods=xd)

    vma = pd.Series(vma).fillna(method='bfill')
    line = Line(vma, 'vwma')
    return line
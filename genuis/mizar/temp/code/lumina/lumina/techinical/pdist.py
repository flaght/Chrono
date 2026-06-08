# -*- encoding:utf-8 -*-
"""
    计算线PDIST模块
"""
import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean
import pdb

def calc_pdist(kl_pd, drift):
    pdist = (kl_pd['high'] - kl_pd['low']).fillna(0)
    pdist += (kl_pd['open'] - kl_pd['close'].shift(1)).abs()
    pdist -= (kl_pd['close'] - kl_pd['open']).abs()

    pdist = pd.Series(pdist).fillna(method='bfill')
    return Line(pdist, 'pdist')
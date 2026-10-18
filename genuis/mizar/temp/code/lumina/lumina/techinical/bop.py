# -*- encoding:utf-8 -*-
"""
    计算线bop模块
"""
import pdb
import pandas as pd
from ultron.ump.technical.line import Line

def calc_bop(kl_pd, scalar=1):
    bop = (kl_pd['close'] - kl_pd['open']) / (kl_pd['high'] - kl_pd['low']) * scalar
    bop = pd.Series(bop).fillna(method='bfill')
    line = Line(bop, 'bop')
    return line
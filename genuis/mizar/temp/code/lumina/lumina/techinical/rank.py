# -*- encoding:utf-8 -*-
"""
    计算rank模块
"""
import numpy as np
import pandas as pd
from ultron.ump.core.helper import pd_rolling_mean,pd_ewm_mean
from ultron.ump.technical.line import Line

def calc_rank(kl_pd, xd=1):
    close = kl_pd['close']
    rank = close.rank()[-1] / close.rank().shape[0]

    rank = pd.Series(rank).fillna(method='bfill')
    line = Line(rank, 'rank')
    return line
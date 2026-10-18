# -*- encoding:utf-8 -*-
"""
    计算线PVI模块
"""
import pandas as pd
import numpy as np
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean
import pdb

def calc_pvi(kl_pd, xd, offset=None, scalar=100):

    xd = int(xd) if xd and xd > 0 else 1
    offset = int(offset) if isinstance(offset, int) else 0

    close = kl_pd['close']
    volume = kl_pd['volume']

    mom = close.diff(xd)
    roc = scalar * mom / close.shift(xd)

    sign = volume.diff(1)
    sign[sign > 0] = 1
    sign[sign < 0] = -1
    sign.iloc[0] = 1

 
    #pvi = sign[sign > 0] * roc
    pvi = sign.where(sign > 0, np.nan) * roc
    ## 先用前一个值填充当前表达PVI无变化，再用后一个值填充前面的nan，用于历史表现
    pvi = pd.Series(pvi).fillna(method='ffill').fillna(method='bfill')
    line = Line(pvi, 'pvi')
    return line
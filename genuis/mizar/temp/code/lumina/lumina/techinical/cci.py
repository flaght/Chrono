# -*- encoding:utf-8 -*-
"""
    计算线cci模块
"""
import pdb
import numpy as np
import pandas as pd
from ultron.ump.core.helper import pd_ewm_std, pd_rolling_std, pd_rolling_mean
from ultron.ump.technical.line import Line


def calc_cci(kl_pd, xd=14, ewm=True):
    high = kl_pd['high']
    low = kl_pd['low']
    close = kl_pd['close']
    tp = (high + low + close) / 3
    if ewm:
        roll_std = pd_ewm_std(tp, span=xd, min_periods=1,
                              adjust=True)
    else:
        roll_std = pd_rolling_std(
            tp, window=xd, min_periods=1, center=False)

    # min_periods=1还是会有两个nan，填了
    roll_std = pd.Series(roll_std).fillna(method='bfill')
    cci = (tp - pd_rolling_mean(tp, window=xd, min_periods=1)) / roll_std
    line = Line(cci, 'cci')
    return line
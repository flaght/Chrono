# -*- encoding:utf-8 -*-
"""
    计算线volatility1模块
"""
import pandas as pd
import numpy as np
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_rolling_std, pd_rolling_sum, pd_ewm_mean, pd_ewm_std, pd_ewm_sum
import pdb


def calc_volatility1(kl_pd, xd, drift, ewm=True):
    xd = int(xd) if xd and xd > 0 else 10
    drift = int(drift) if drift and drift > 0 else 1

    close = kl_pd['close']
    close[close < 0] = np.nan

    if ewm:
        ma = pd_ewm_mean(close, span=xd, min_periods=1)
        ma[ma <= 0] = np.nan
        std = pd_ewm_std(close, span=xd, min_periods=1)
        std[std <= 0] = np.nan
    else:
        ma = pd_rolling_mean(close, window=xd, min_periods=1)
        ma[ma <= 0] = np.nan
        std = pd_rolling_std(close, window=xd, min_periods=1)
        std[std <= 0] = np.nan

    boll_cls2up = close / (ma + 2 * std) - 1
    boll_cls2dow = close / (ma - 2 * std) - 1

    boll_cls2up[np.isinf(boll_cls2up)] = np.nan
    boll_cls2dow[np.isinf(boll_cls2dow)] = np.nan

    boll_cls2up = pd.Series(boll_cls2up).fillna(method='bfill')
    boll_cls2dow = pd.Series(boll_cls2dow).fillna(method='bfill')

    if ewm:
        boll_cls2up = pd_ewm_sum(boll_cls2up, span=drift, min_periods=1)
        boll_cls2dow = pd_ewm_sum(boll_cls2dow, span=drift, min_periods=1)
    else:
        boll_cls2up = pd_rolling_sum(boll_cls2up, window=drift, min_periods=1)
        boll_cls2dow = pd_rolling_sum(boll_cls2dow,
                                      window=drift,
                                      min_periods=1)

    boll_cls2dow = pd.Series(boll_cls2dow).fillna(method='bfill')
    boll_cls2up = pd.Series(boll_cls2up).fillna(method='bfill')

    return Line(boll_cls2up, 'boll_cls2up'), Line(boll_cls2dow, 'boll_cls2dow')

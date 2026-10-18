# -*- encoding:utf-8 -*-
"""
    #左右概率差
#右端收益VaR - 左端收益VaR
"""

import pdb
import pandas as pd
import numpy as np
from scipy.stats import norm
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean, pd_rolling_std, pd_ewm_std


def calc_extvar(kl_pd, xd, ewm=True):
    ret = np.log(kl_pd['close'] / kl_pd['close'].shift(1))
    if ewm:
        mean_ret = pd_ewm_mean(ret, span=xd, min_periods=1)
        mean_std = pd_ewm_std(ret, span=xd, min_periods=1)
    else:
        mean_ret = pd_rolling_mean(ret, window=xd, min_periods=1)
        mean_std = pd_rolling_std(ret, window=xd, min_periods=1)

    ext_big = mean_ret + mean_std + norm.ppf(1 - 0.01)
    ext_small = -mean_ret + mean_std + norm.ppf(1 - 0.01)

    if ewm:
        maximum = pd_ewm_mean(pd.Series(np.where(ret > ext_big, ret, np.nan)),
                              span=xd,
                              min_periods=1)
        minimum_mean = pd_ewm_mean(pd.Series(
            np.where(ret < ext_small, ret, np.nan)),
                                   span=xd,
                                   min_periods=1)
    else:
        maximum = pd_rolling_mean(pd.Series(
            np.where(ret > ext_big, ret, np.nan)),
                                  window=xd,
                                  min_periods=1)
        minimum_mean = pd_rolling_mean(pd.Series(
            np.where(ret < ext_small, ret, np.nan)),
                                       window=xd,
                                       min_periods=1)

    line = Line(
        maximum - minimum_mean,
        'extvar') if not maximum.empty and not minimum_mean.empty else Line(
            0, 'extvar')
    return line

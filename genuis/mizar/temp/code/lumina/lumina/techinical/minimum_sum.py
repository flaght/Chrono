# -*- encoding:utf-8 -*-
"""
    收益率极大值的绝对幅度均值
"""
import pdb
import pandas as pd
import numpy as np
from scipy.stats import norm
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean, pd_rolling_std, pd_ewm_std, pd_ewm_sum, pd_rolling_sum


def calc_minimum_sum(kl_pd, xd, ewm=True):
    ret = np.log(kl_pd['close'] / kl_pd['close'].shift(1))
    if ewm:
        mean_ret = pd_ewm_mean(ret, span=xd, min_periods=1)
        mean_std = pd_ewm_std(ret, span=xd, min_periods=1)
    else:
        mean_ret = pd_rolling_mean(ret, window=xd, min_periods=1)
        mean_std = pd_rolling_std(ret, window=xd, min_periods=1)

    ext_small = -mean_ret + mean_std * norm.ppf(1 - 0.01)

    if ewm:
        minimum = pd_ewm_sum(pd.Series(np.where(ret < ext_small, ret, np.nan)),
                             span=xd,
                             min_periods=1)
    else:
        minimum = pd_rolling_sum(pd.Series(
            np.where(ret < ext_small, ret, np.nan)),
                                 window=xd,
                                 min_periods=1)

    line = Line(minimum, 'minimum_freq') if not minimum.empty else Line(
        0, 'minimum_freq')
    return line

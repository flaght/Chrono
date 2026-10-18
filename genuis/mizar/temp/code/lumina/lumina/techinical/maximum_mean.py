# -*- encoding:utf-8 -*-
"""
    收益率极大值的绝对幅度均值
"""
import pdb
import pandas as pd
import numpy as np
from scipy.stats import norm
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean, pd_rolling_std, pd_ewm_std


def calc_maximum_mean(kl_pd, xd, ewm=True):
    ret = np.log(kl_pd['close'] / kl_pd['close'].shift(1))
    if ewm:
        mean_ret = pd_ewm_mean(ret, span=xd, min_periods=1)
        mean_std = pd_ewm_std(ret, span=xd, min_periods=1)
    else:
        mean_ret = pd_rolling_mean(ret, window=xd, min_periods=1)
        mean_std = pd_rolling_std(ret, window=xd, min_periods=1)

    ext_big = mean_ret + mean_std * norm.ppf(1 - 0.01)

    if ewm:
        maximum = pd_ewm_mean(pd.Series(np.where(ret > ext_big, ret, np.nan)), span=xd, min_periods=1)
    else:
        maximum = pd_rolling_mean(pd.Series(np.where(ret > ext_big, ret, np.nan)), window=xd, min_periods=1)
    
    line = Line(maximum, 'maximum_mean') if not maximum.empty else Line(
        0, 'maximum_mean')
    return line

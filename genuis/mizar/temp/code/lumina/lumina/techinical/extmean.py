# -*- encoding:utf-8 -*-
"""
#左右收益差
#出现大于右端收益率平均值 - 出现小于左端收益率平均值

"""

import pdb
import pandas as pd
import numpy as np
from scipy.stats import  norm
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean, pd_rolling_std, pd_ewm_std

def calc_extmean(kl_pd, xd, ewm=True):
    ret = np.log(kl_pd['close'] / kl_pd['close'].shift(1))
    if ewm:
        mean_ret = pd_ewm_mean(ret, span=xd, min_periods=1)
        mean_std  = pd_ewm_std(ret, span=xd, min_periods=1)
    else:
        mean_ret = pd_rolling_mean(ret, window=xd, min_periods=1)
        mean_std = pd_rolling_std(ret, window=xd, min_periods=1)

    ext_big = mean_ret + mean_std + norm.ppf(1 - 0.01)
    ext_small = -mean_ret + mean_std + norm.ppf(1 - 0.01)

    ext_com = pd.Series(np.where((ret > ext_big) | (ret < ext_small), ret, np.nan))
    if ewm:
        extmean = pd_ewm_mean(ext_com, span=xd, min_periods=1)
    else:
        extmean = pd_rolling_mean(ext_com, window=xd, min_periods=1)

    line = Line(extmean, 'extmean') if not extmean.empty else Line(0, 'extmean')
    return line

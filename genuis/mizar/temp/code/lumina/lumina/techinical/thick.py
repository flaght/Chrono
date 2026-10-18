# -*- encoding:utf-8 -*-
"""
    对数成交量平均厚尾分布
"""
import pdb
import pandas as pd
import numpy as np
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean, pd_rolling_sum, pd_ewm_sum


def calc_thick(kl_pd, xd, quant=0.1, direction=True, ewm=True):
    vol = kl_pd['volume'] + 1e-10
    vol = vol.apply(lambda x: np.log(x))

    if ewm:
        thick = pd_ewm_mean(vol, span=xd, min_periods=1)
    else:
        thick = pd_rolling_mean(vol, window=xd, min_periods=1)

    mask_data = thick.mask(
        thick.sub(thick.quantile(quant)) > 0) if direction else thick.mask(
            thick.sub(thick.quantile(quant)) < 0)
    thick = mask_data.where(np.isnan(mask_data), 1) * thick

    if ewm:
        thick = pd_ewm_sum(thick, span=xd, min_periods=1)
    else:
        thick = pd_rolling_sum(thick, window=xd, min_periods=1)

    thick = pd.Series(thick).fillna(method='bfill')
    line = Line(thick, 'thick')
    return line

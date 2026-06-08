# -*- encoding:utf-8 -*-
"""
    计算线EFI模块
"""
import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean
import pdb


def calc_efi(kl_pd, xd, drift, ewm=False):

    xd = int(xd) if xd and xd > 0 else 13
    
    drift = int(drift) if isinstance(drift, int) and drift != 0 else 1

    pv_diff = kl_pd['close'].diff(drift) * kl_pd['volume']

    if ewm:
        efi = pd_ewm_mean(pv_diff, span=xd, min_periods=xd)
    else:
        efi = pd_rolling_mean(pv_diff, window=xd, min_periods=xd)

    efi = pd.Series(efi).fillna(method='bfill')
    line = Line(efi, 'efi')
    return line

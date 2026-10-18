# -*- encoding:utf-8 -*-
"""
    计算线volatility2模块
"""
import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean
import pdb


def calc_volatility2(kl_pd, fast, low, fdrift, sdrift, ewm=True):
    vol = kl_pd['volume']
    fast = int(fast) if fast and fast > 0 else 20
    low = int(low) if low and low > 0 else 120
    fdrift = int(fdrift) if fdrift and fdrift > 0 else 10
    sdrift = int(sdrift) if sdrift and sdrift > 0 else 30

    if ewm:
        stdf = pd_ewm_mean(vol, span=fast, min_periods=fdrift)
        stds = pd_ewm_mean(vol, span=low, min_periods=sdrift)
    else:
        stdf = pd_rolling_mean(vol, window=fast, min_periods=fdrift)
        stds = pd_rolling_mean(vol, window=low, min_periods=sdrift)

    volatility = -(stdf / stds)
    volatility = pd.Series(volatility).fillna(method='bfill')
    line = Line(volatility, 'volatility2')
    return line

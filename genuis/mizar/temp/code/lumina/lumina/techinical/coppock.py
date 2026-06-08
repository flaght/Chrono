# -*- encoding:utf-8 -*-
"""
    计算线coppock模块
"""
import pdb
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean


def calc_coppock(kl_pd, fast, slow, xd, scalar, drift, ewm=True):

    xd = int(xd) if xd and xd > 0 else 10
    fast = int(fast) if fast and fast > 0 else 11
    slow = int(slow) if slow and slow > 0 else 14
    ro = scalar * (kl_pd['close'] -
                   kl_pd['close'].shift(drift)) / kl_pd['close'].shift(drift)

    if ewm:
        roc = pd_ewm_mean(ro, span=fast, min_periods=fast) + pd_ewm_mean(
            ro, span=slow, min_periods=slow)
        coppock = pd_ewm_mean(roc, span=xd, min_periods=xd)
    else:
        roc = pd_rolling_mean(ro, window=fast,
                              min_periods=fast) + pd_rolling_mean(
                                  ro, window=slow, min_periods=slow)
        coppock = pd_rolling_mean(roc, window=xd, min_periods=xd)

    line = Line(coppock, 'coppock')
    return line

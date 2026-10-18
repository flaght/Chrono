# -*- encoding:utf-8 -*-
"""
    计算线UI模块
"""

import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_max, pd_rolling_mean, pd_ewm_mean
import pdb


def calc_ui(kl_pd, xd, scalar, ewm=True):

    xd = int(xd) if xd and xd > 0 else 14
    scalar = float(scalar) if scalar and scalar > 0 else 100

    close = kl_pd['close']
    high = kl_pd['high']

    high_close = pd_rolling_max(high, window=xd, min_periods=xd)
    downside = scalar * (close - high_close)
    downside = downside / high_close
    d2 = downside**2
    if ewm:
        ui = pd_ewm_mean(d2, span=xd, min_periods=xd)
    else:
        ui = pd_rolling_mean(d2, window=xd, min_periods=xd)

    ui = pd.Series(ui).fillna(method='bfill')
    line = Line(ui, 'ui')
    return line

# -*- encoding:utf-8 -*-
"""
    计算线ao模块
"""
import numpy as np
import pandas as pd
from ultron.kdutils import regression
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import  pd_rolling_mean, pd_ewm_mean


def calc_angle(kl_pd, xd,  ewm=False):
    close = kl_pd['close']
    if ewm:
        ma = pd_ewm_mean(close, span=xd, min_periods=xd)
    else:
        ma = pd_rolling_mean(close, window=xd, min_periods=xd)
    _, angle1 = regression.regress_y(ma.dropna(), mode=True, zoom=True)
    angle = np.full_like(close.values, fill_value=np.nan, dtype=float)
    angle[-len(angle1):] = angle1
    line = Line(angle, 'angle')
    return line
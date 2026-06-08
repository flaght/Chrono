# -*- encoding:utf-8 -*-
"""
    计算线TrueRange模块
"""
import pandas as pd
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean
import pdb


def trun_range(kl_pd, drift):
    high = kl_pd['high']
    low = kl_pd['low']

    prev_close = kl_pd['close'].shift(1)
    high_low_range = (high - low).fillna(0)
    true_range = high_low_range
    true_range[true_range < high - prev_close] = high - prev_close
    true_range[true_range < prev_close - low] = prev_close - low
    true_range.iloc[:drift] = 0

    true_range = pd.Series(true_range).fillna(method='bfill')

    line = Line(true_range, 'true_range')
    return line

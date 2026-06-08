"""
cr045：多窗口极差比值的滑动窗口排序分位因子，衡量多尺度极端波动。
计算方式：短期极差与长期极差之比，在N期内排序分位，滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr045(high, low, slow, fast, weriod, window, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    #high.rolling(window).max() - low.rolling(window).min()
    short_range = roller_max(high, window=fast,
                                 min_periods=1) - roller_min(
                                     low, window=fast, min_periods=1)
    long_range = roller_max(
        high, window=slow, min_periods=1) - roller_min(
            low, window=slow, min_periods=1
        )  #high.rolling(weriod).max() - low.rolling(weriod).min()
    ratio = short_range / (long_range + 1e-8)

    def rolling_rank(x):
        return pd.Series(x).rank(pct=True).iloc[-1]

    rank_quant = ratio.rolling(weriod).apply(rolling_rank, raw=False)
    factor = roller_mean(rank_quant, window, 1, method)
    return factor

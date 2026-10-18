"""
cr050：多窗口持仓量(openint)极差比值的滑动窗口排序分位因子，衡量多尺度持仓极端波动。
计算方式：短期持仓量极差与长期持仓量极差之比，在N期内排序分位，滑动平均。
本因子为cr045的持仓量(openint)版本。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *

def cr050(openint, slow, fast, weriod, window, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    short_range = roller_max(openint, fast, 1, 'rolling') - roller_min(openint, fast, 1, 'rolling')
    long_range = roller_max(openint, slow, 1, 'rolling') - roller_min(openint, slow, 1, 'rolling')
    ratio = short_range / (long_range + 1e-8)
    def rolling_rank(x):
        return pd.Series(x).rank(pct=True).iloc[-1]
    rank_quant = ratio.rolling(weriod).apply(rolling_rank, raw=False)
    factor = roller_mean(rank_quant, window, 1, method)
    return factor 
"""
cr036：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合移动窗口排序分位因子，衡量高阶排序极端性。
计算方式：三变量中心化后乘积在N期内排序，取分位排名（如90%），再做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr036(close, high, low, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    log_ret = np.log(close / close.shift(1))
    #price_range = high.rolling(weriod).max() - low.rolling(weriod).min()
    price_range = roller_max(high, window=weriod,
                                 min_periods=1) - roller_min(
                                     low, window=weriod, min_periods=1)
    vol_chg = volume.pct_change()
    #log_ret_c = log_ret - log_ret.rolling(weriod).mean()
    log_ret_c = log_ret - roller_mean(log_ret, weriod, 1, method)
    #price_range_c = price_range - price_range.rolling(weriod).mean()
    price_range_c = price_range - roller_mean(price_range, weriod, 1, method)
    #vol_chg_c = vol_chg - vol_chg.rolling(weriod).mean()
    vol_chg_c = vol_chg - roller_mean(vol_chg, weriod, 1, method)
    mix_prod = log_ret_c * price_range_c * vol_chg_c

    # 排序分位（rolling窗口内排名/窗口长度）
    def rolling_rank(x):
        return pd.Series(x).rank(pct=True).iloc[-1]

    rank_quant = mix_prod.rolling(weriod).apply(rolling_rank, raw=False)
    factor = roller_mean(rank_quant, window, 1, method)
    return factor

"""
cr034：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合极端值复合因子，衡量收益、极端波动与量能变化的高阶极端风险。
计算方式：先计算N期收盘价对数收益率、N期最高-最低极差、N期成交量变化率的三阶混合极端值（如最大/最小），再做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr034(close, high, low, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    log_ret = np.log(close / close.shift(1))
    #price_range = high.rolling(weriod).max() - low.rolling(weriod).min()
    price_range = roller_max(high, window=weriod,
                                 min_periods=1) - roller_min(
                                     low, window=weriod, min_periods=1)
    vol_chg = volume.pct_change()
    # 三阶混合极端值（近似实现：三变量中心化后乘积的rolling最大/最小）
    #log_ret_c = log_ret - log_ret.rolling(weriod).mean()
    log_ret_c = log_ret - roller_mean(log_ret, weriod, 1, method)
    #price_range_c = price_range - price_range.rolling(weriod).mean()
    price_range_c = price_range - roller_mean(price_range, weriod, 1, method)
    #vol_chg_c = vol_chg - vol_chg.rolling(weriod).mean()
    vol_chg_c = vol_chg - roller_mean(vol_chg, weriod, 1, method)
    mix_prod = log_ret_c * price_range_c * vol_chg_c
    #mix_max = mix_prod.rolling(weriod).max()
    #mix_min = mix_prod.rolling(weriod).min()
    mix_max = roller_max(mix_prod, weriod, 1, 'rolling')
    mix_min = roller_min(mix_prod, weriod, 1, 'rolling')
    factor = roller_mean(mix_max - mix_min, window, 1, method)
    return factor

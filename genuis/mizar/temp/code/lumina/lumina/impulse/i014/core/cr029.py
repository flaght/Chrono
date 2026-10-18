"""
cr029：N期收盘价对数收益率、最高价极差、成交量变化率三者的三阶混合峰度复合因子，衡量收益、极端波动与量能变化的高阶陡峭性。
计算方式：先计算N期收盘价对数收益率、N期最高-最低极差、N期成交量变化率的三阶混合峰度，再做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr029(close, high, low, volume, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    log_ret = np.log(close / close.shift(1))
    #price_range = high.rolling(weriod).max() - low.rolling(weriod).min()
    price_range = roller_max(high, window=weriod,
                                 min_periods=1) - roller_min(
                                     low, window=weriod, min_periods=1)
    vol_chg = volume.pct_change()
    # 三阶混合峰度（近似实现：四变量中心化后乘积的均值/标准差四次方）
    #log_ret_c = log_ret - log_ret.rolling(weriod).mean()
    log_ret_c = log_ret - roller_mean(log_ret, weriod, 1, method)
    #price_range_c = price_range - price_range.rolling(weriod).mean()
    price_range_c = price_range - roller_mean(price_range, weriod, 1, method)
    #vol_chg_c = vol_chg - vol_chg.rolling(weriod).mean()
    vol_chg_c = vol_chg - roller_mean(vol_chg, weriod, 1, method)
    #std_prod = (log_ret.rolling(weriod).std() *
    #            price_range.rolling(weriod).std() *
    #            vol_chg.rolling(weriod).std())

    std_prod = roller_std(log_ret, weriod, 1, method) * roller_std(
        price_range, weriod, 1, method) * roller_std(vol_chg, weriod, 1,
                                                     method)
    #mix_kurt = (log_ret_c * price_range_c *
    #            vol_chg_c**2).rolling(weriod).mean() / (std_prod**2 + 1e-8)
    mix_kurt = roller_mean(log_ret_c * price_range_c * vol_chg_c**2, weriod, 1,
                           method) / (std_prod**2 + 1e-8)
    factor = roller_mean(mix_kurt, window, 1, method)
    return factor

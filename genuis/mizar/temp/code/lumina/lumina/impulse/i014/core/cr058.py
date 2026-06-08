"""
cr058：N期收盘价对数收益率、最高价极差、持仓量(openint)变化率三者的三阶混合移动窗口排序分位因子，衡量高阶持仓排序极端性。
计算方式：三变量中心化后乘积在N期内排序，取分位排名（如90%），再做滑动平均。
本因子为cr036的持仓量(openint)版本。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *

def cr058(close, high, low, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    log_ret = np.log(close / close.shift(1))
    price_range = roller_max(high, weriod, 1, 'rolling') - roller_min(low, weriod, 1, 'rolling')
    oi_chg = openint.pct_change()
    log_ret_c = log_ret - roller_mean(log_ret, weriod, 1, method)
    price_range_c = price_range - roller_mean(price_range, weriod, 1, method)
    oi_chg_c = oi_chg - roller_mean(oi_chg, weriod, 1, method)
    mix_prod = log_ret_c * price_range_c * oi_chg_c
    def rolling_rank(x):
        return pd.Series(x).rank(pct=True).iloc[-1]
    rank_quant = mix_prod.rolling(weriod).apply(rolling_rank, raw=False)
    factor = roller_mean(rank_quant, window, 1, method)
    return factor 
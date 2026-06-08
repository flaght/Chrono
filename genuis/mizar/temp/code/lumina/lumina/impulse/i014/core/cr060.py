"""
cr060：N期收盘价对数收益率、最高价极差、持仓量(openint)变化率三者的三阶混合极端值复合因子，衡量收益、极端波动与持仓变化的高阶极端风险。
计算方式：先计算N期收盘价对数收益率、N期最高-最低极差、N期持仓量变化率的三阶混合极端值（如最大/最小），再做滑动平均。
本因子为cr034的持仓量(openint)版本。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *

def cr060(close, high, low, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    log_ret = np.log(close / close.shift(1))
    price_range = roller_max(high, weriod, 1, 'rolling') - roller_min(low, weriod, 1, 'rolling')
    oi_chg = openint.pct_change()
    log_ret_c = log_ret - roller_mean(log_ret, weriod, 1, method)
    price_range_c = price_range - roller_mean(price_range, weriod, 1, method)
    oi_chg_c = oi_chg - roller_mean(oi_chg, weriod, 1, method)
    mix_prod = log_ret_c * price_range_c * oi_chg_c
    mix_max = roller_max(mix_prod, weriod, 1, 'rolling')
    mix_min = roller_min(mix_prod, weriod, 1, 'rolling')
    factor = roller_mean(mix_max - mix_min, window, 1, method)
    return factor 
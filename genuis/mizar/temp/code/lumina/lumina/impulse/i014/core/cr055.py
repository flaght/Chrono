"""
cr055：N期收盘价对数收益率、最高价极差、持仓量(openint)变化率三者的三阶混合移动窗口绝对值均值因子，衡量高阶持仓绝对波动性。
计算方式：三变量中心化后乘积在N期内绝对值均值，滑动平均。
本因子为cr039的持仓量(openint)版本。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *

def cr055(close, high, low, openint, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    log_ret = np.log(close / close.shift(1))
    price_range = roller_max(high, weriod, 1, 'rolling') - roller_min(low, weriod, 1, 'rolling')
    oi_chg = openint.pct_change()
    log_ret_c = log_ret - roller_mean(log_ret, weriod, 1, method)
    price_range_c = price_range - roller_mean(price_range, weriod, 1, method)
    oi_chg_c = oi_chg - roller_mean(oi_chg, weriod, 1, method)
    mix_prod = log_ret_c * price_range_c * oi_chg_c
    abs_mean = mix_prod.rolling(weriod).apply(lambda x: np.mean(np.abs(x)), raw=True)
    factor = roller_mean(abs_mean, window, 1, method)
    return factor 
"""
cr007：N期收盘价动量与波动率复合因子，衡量价格趋势与风险的综合效应。
计算方式：先计算N期动量（收盘价/前N期收盘价-1），再计算N期收益率标准差，两者归一化后相乘，最后做滑动平均。
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def cr007(close, window, weriod, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    momentum = close / close.shift(weriod) - 1
    vol = roller_std(safe_log(close, 1), weriod, 1, method)
    # 归一化
    norm_mom = (momentum - roller_mean(momentum, weriod, 1, method)) / (
        roller_std(momentum, weriod, 1, method) + 1e-8)

    norm_vol = (vol - roller_mean(vol, weriod, 1, method)) / (
        roller_std(vol, weriod, 1, method) + 1e-8)
    factor = norm_mom * norm_vol
    factor = roller_mean(factor, window, 1, method)
    return factor

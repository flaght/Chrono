# -*- encoding:utf-8 -*-
import numpy as np
from lumina.impulse.fixed import *

def ki009(open, high, low, close, window, weriod, ewm=False):
    """
    Dual Thrust 双向突破因子

    计算逻辑:
    1. Range = Max(HH-LC, HC-LL)
       - HH = N日最高价的最高
       - LC = N日收盘价的最低
       - HC = N日收盘价的最高
       - LL = N日最低价的最低
    2. 上轨 = Open + K1 * Range
    3. 下轨 = Open - K2 * Range
    4. 因子 = (Close - 上轨) / Range 当Close > 上轨
            = (Close - 下轨) / Range 当Close < 下轨
            = 0 其他情况

    参数:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        window: 计算Range的窗口期N
        weriod: 最终平滑窗口期
        ewm: 是否使用指数加权移动平均

    返回:
        alpha: Dual Thrust突破因子值
    """
    method = 'ewm' if ewm else 'rolling'

    # K1和K2系数，通常设为0.5-0.7之间，这里使用0.6
    k1 = 0.6
    k2 = 0.6

    # 计算HH, LL, HC, LC
    hh = roller_max(high, weriod, weriod, 'rolling')  # N日最高价的最高
    ll = roller_min(low, weriod, weriod, 'rolling')   # N日最低价的最低
    hc = roller_max(close, weriod, weriod, 'rolling') # N日收盘价的最高
    lc = roller_min(close, weriod, weriod, 'rolling') # N日收盘价的最低

    # 计算Range = Max(HH-LC, HC-LL)
    range1 = hh - lc
    range2 = hc - ll
    range_val = np.maximum(range1, range2)

    # 避免除零
    range_val = np.where(range_val == 0, np.nan, range_val)

    # 计算上轨和下轨
    upper_band = open + k1 * range_val
    lower_band = open - k2 * range_val

    # 计算因子值
    # 当收盘价突破上轨时，信号为正
    upper_signal = (close - upper_band) / range_val
    # 当收盘价突破下轨时，信号为负
    lower_signal = (close - lower_band) / range_val

    # 组合信号：突破上轨用上轨信号，突破下轨用下轨信号，否则为0
    #factor = np.where(close > upper_band, upper_signal,
    #                 np.where(close < lower_band, lower_signal, 0))
    factor = upper_signal.where(close > upper_band,
                               lower_signal.where(close < lower_band, 0))

    # 最终平滑
    alpha = roller_mean(factor, window, window, method)

    return alpha

# -*- encoding:utf-8 -*-
import numpy as np
from lumina.impulse.fixed import *

def ki010(open, high, low, close, window, weriod, ewm=False):
    """
    ATR突破因子

    计算逻辑:
    1. True Range = Max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    2. ATR = N日True Range均值
    3. 因子 = (Close - Open) / ATR，衡量日内动量相对波动率

    参数:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        window: 计算ATR的窗口期N
        weriod: 最终平滑窗口期
        ewm: 是否使用指数加权移动平均

    返回:
        alpha: ATR突破因子值
    """
    method = 'ewm' if ewm else 'rolling'

    # 计算前一日收盘价
    prev_close = close.shift(1)

    # 计算True Range的三个组成部分
    tr1 = high - low                    # 当日最高最低价差
    tr2 = np.abs(high - prev_close)     # 当日最高价与昨收价差
    tr3 = np.abs(low - prev_close)      # 当日最低价与昨收价差

    # True Range取三者最大值
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))

    # 计算ATR：N日True Range均值
    atr = roller_mean(true_range, weriod, weriod, method)

    # 避免除零
    atr = np.where(atr == 0, np.nan, atr)

    # 计算日内动量相对ATR的比率
    # 正值表示上涨动量，负值表示下跌动量
    intraday_momentum = close - open
    factor = intraday_momentum / atr

    # 最终平滑
    alpha = roller_mean(factor, window, window, method)

    return alpha

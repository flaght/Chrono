import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def ki018(close, high, low, volume, fast, slow, window, ewm=False):
    """
    多空对比因子 (Bull-Bear Ratio)

    来源: 华西证券《基于量价因子的ETF组合策略》

    原理:
        通过日内K线位置判断多空力量对比
        (Close - Low) 代表多头力量
        (High - Close) 代表空头力量
        使用成交量加权，放大大成交量时段的信号

    公式:
        ratio = Volume × [(Close - Low) - (High - Close)] / (High - Low)
        factor = EWMA(ratio, window1) - EWMA(ratio, window2)

    参数:
        close: 收盘价 DataFrame
        high: 最高价 DataFrame
        low: 最低价 DataFrame
        volume: 成交量 DataFrame
        window: 外层平滑窗口
        weriod: 日内周期

    返回:
        多空对比因子值

    信号解读:
        > 0: 多头占优
        < 0: 空头占优
    """
    method = 'ewm' if ewm else 'rolling'

    # 计算多空力量对比
    bull_power = close - low  # 多头力量
    bear_power = high - close  # 空头力量

    # 防止除零
    price_range = high - low
    price_range = price_range.replace(0, np.nan)

    # 量加权的多空比率
    bull_bear_ratio = volume * (bull_power - bear_power) / price_range

    # 短期和长期均值
    short_window = fast
    long_window = slow * 3

    short_ma = roller_mean(bull_bear_ratio, short_window, short_window, method)
    long_ma = roller_mean(bull_bear_ratio, long_window, long_window, method)

    # 多空对比变化
    alpha = short_ma - long_ma

    return alpha

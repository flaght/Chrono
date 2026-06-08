import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def ki017(close, volume, openint, window, weriod, ewm=False):
    """
    量价背离因子 (Price-Volume Divergence)

    来源: 华西证券《基于量价因子的ETF组合策略》

    原理:
        计算价格与成交量的负相关性
        量价背离表示价格上涨但成交量萎缩，或价格下跌但成交量放大
        负相关越强，后续反转概率越高

    公式:
        factor = -correlation(Close, Volume, window)

    参数:
        close: 收盘价 DataFrame (分钟级)
        volume: 成交量 DataFrame (分钟级)
        openint: 持仓量 DataFrame (分钟级)
        window: 外层平滑窗口
        weriod: 日内周期 (如240分钟/天)
        ewm: 是否使用指数加权

    返回:
        量价背离因子值

    信号解读:
        > 0: 量价背离，看跌信号
        < 0: 量价同向，趋势延续
    """
    method = 'ewm' if ewm else 'rolling'
    #pdb.set_trace()
    # 计算日内价格和成交量
    daily_close = roller_mean(close, weriod, weriod, method)
    daily_volume = roller_sum(volume, weriod, weriod, method)

    # 计算价格与成交量的相关系数
    pv_corr = roller_corr(daily_close, daily_volume, weriod, weriod, method)

    # 取负值 - 负相关表示量价背离
    alpha = -pv_corr

    # 平滑处理
    alpha = roller_mean(alpha, window, window, method)

    return alpha

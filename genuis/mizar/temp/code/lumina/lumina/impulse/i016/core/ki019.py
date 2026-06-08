import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def ki019(close, window, fast, slow, ewm=False):
    """
    动量期限差因子 (Momentum Term Spread)

    来源: 华西证券《基于量价因子的ETF组合策略》

    原理:
        计算短期动量与长期动量的差值
        捕捉动量加速或减速的信号
        动量期限差为正表示短期动量强于长期，趋势加速
        动量期限差为负表示短期动量弱于长期，趋势减速

    公式:
        mom_short = (Close_t - Close_{t-window1}) / Close_{t-window1}
        mom_long = (Close_t - Close_{t-window2}) / Close_{t-window2}
        factor = mom_short - mom_long

    参数:
        close: 收盘价 DataFrame
        volume: 成交量 DataFrame (未使用)
        openint: 持仓量 DataFrame (未使用)
        window: 短期窗口
        weriod: 日内周期，长期窗口 = weriod

    返回:
        动量期限差因子值

    信号解读:
        > 0: 趋势加速，动量增强
        < 0: 趋势减速，动量减弱
    """
    method = 'ewm' if ewm else 'rolling'

    # 短期动量 (window周期)
    close_lag_short = close.shift(fast)
    mom_short = (close - close_lag_short) / (close_lag_short + 1e-10)

    # 长期动量 (weriod周期)
    close_lag_long = close.shift(slow)
    mom_long = (close - close_lag_long) / (close_lag_long + 1e-10)

    # 动量期限差
    mom_spread = mom_short - mom_long

    # 平滑处理
    alpha = roller_mean(mom_spread, window, window, method)

    return alpha

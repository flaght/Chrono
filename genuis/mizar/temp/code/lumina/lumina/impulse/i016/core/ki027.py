# -*- encoding:utf-8 -*-
"""
加权上影线频率因子 (西南证券)
来源: 20250823-西南证券-因子选股系列：加权影线频率与K线形态因子.pdf

公式:
加权上影线频率 = Σ(w_j × 1_{上影线>u}) / M
其中:
- 上影线 = (最高价 - max(开盘价, 收盘价)) / 前收盘价
- 阈值 u = 1%
- 回望期 M = weriod 个交易日
- w_j = 0.5^((t-j)/λ) 为衰退权重
"""
from lumina.impulse.fixed import *


def ki027(close, open, high, window, weriod, threshold=0.01, ewm=False):
    """
    加权上影线频率因子

    参数:
        close: 收盘价序列
        open_: 开盘价序列
        high: 最高价序列
        window: 外层平滑窗口
        weriod: 回望周期
        ewm: 是否使用指数加权

    返回:
        alpha: 因子值
    """
    method = 'ewm' if ewm else 'rolling'

    # 计算上影线比例
    # 上影线 = (最高价 - max(开盘价, 收盘价)) / 前收盘价
    
    upper_body = np.maximum(close, open)
    upper_shadow = (high - upper_body) / close.shift(1)

    # 阈值判断 (u = 1%)
    signal = (upper_shadow > threshold).astype(float)

    # 衰退加权求和 (使用 ewm 方法模拟衰退权重)
    weighted_sum = roller_sum(signal, weriod, weriod, 'ewm')

    # 标准化
    core = weighted_sum / weriod
    # 外层平滑
    alpha = roller_mean(core, window, window, method)

    return alpha

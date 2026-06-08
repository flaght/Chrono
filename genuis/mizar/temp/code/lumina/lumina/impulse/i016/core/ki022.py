# -*- encoding:utf-8 -*-
"""
二阶动量因子 (华西证券)
来源: 基于量价因子的ETF组合策略.pdf

公式: 动量的变化量（动量加速度）
含义: 短期动量相对于长期动量的变化
      二阶动量为正：趋势在加速
      二阶动量为负：趋势在减速
"""
from lumina.impulse.fixed import *


def ki022(close, window, fast, slow, ewm=False):
    """
    二阶动量因子

    参数:
        close: 收盘价序列
        window: 外层平滑窗口
        weriod: 回望周期（一阶动量窗口）
        ewm: 是否使用指数加权

    返回:
        alpha: 因子值
    """
    method = 'ewm' if ewm else 'rolling'

    # 计算一阶动量：当前价格相对于均价的偏离
    price_mean = roller_mean(close, slow, slow, method)
    momentum_1 = safe_div(close - price_mean, price_mean)

    # 计算二阶动量：一阶动量的变化
    # 使用较短窗口计算动量变化
    #short_window = max(weriod // 2, 5)
    momentum_2 = momentum_1 - momentum_1.shift(fast)

    # 外层平滑
    alpha = roller_mean(momentum_2, window, window, method)

    return alpha

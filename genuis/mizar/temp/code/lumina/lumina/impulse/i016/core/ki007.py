"""
因子: dz002 - 四分位差立方因子
来源: 东证期货 - 国债期货量价因子挖掘 (2022-07-12)
复现日期: 2026-01-02
"""
from lumina.impulse.fixed import *
import numpy as np


def ki007(high, low, close, window, weriod, ewm=False):
    """
    四分位差立方因子

    研报表达式: cube(X15)
    表现: TS窗口2夏普1.70, TF窗口2夏普1.20
    """
    method = 'ewm' if ewm else 'rolling'

    price_range = high - low
    iqr_proxy = roller_mean(price_range, weriod, weriod, method)

    # 标准化
    iqr_norm = iqr_proxy / (roller_mean(close, weriod, weriod, method) + 1e-8)

    # 立方变换
    iqr_cube = iqr_norm ** 3

    # 最终平滑
    alpha = roller_mean(-iqr_cube, window, window, method)

    return alpha

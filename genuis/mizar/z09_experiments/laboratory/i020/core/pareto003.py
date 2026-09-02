"""
pareto003_core.py — Core 引擎：持仓量分位比率方向性平滑因子
"""
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *


def pareto003(openint, close, window, fast, slow, weriod, ewm=False,
              q_low=0.1, q_high=0.9, epsilon=1e-6, alpha=0.2):
    """
    持仓量分位比率方向性平滑因子

    参数：
    - openint: 持仓量宽表 DataFrame
    - close: 收盘价宽表 DataFrame
    - window: 最终平滑窗口（框架强制）
    - fast: 分位数计算窗口 N
    - slow: 滚动标准化窗口 M
    - weriod: 价格动量窗口 k
    - ewm: 是否使用指数加权（最终平滑）
    - q_low, q_high: 分位数阈值
    - epsilon: 防除零极小值
    - alpha: 指数平滑系数（用于内部ewm）
    """
    method = 'ewm' if ewm else 'rolling'

    # 1. 计算持仓量分位数（仅支持 rolling）
    Q_low = roller_quantile(openint, q_low, fast, 1, 'rolling')
    Q_high = roller_quantile(openint, q_high, fast, 1, 'rolling')

    # 2. 防除零：将 Q_low 中小于 epsilon 的值替换为 epsilon
    Q_low_clipped = Q_low.where(Q_low > epsilon, epsilon)

    # 3. 对数比率 L = log(Q_high / Q_low_clipped)
    L = safe_log(Q_high, Q_low_clipped)

    # 4. 滚动标准化（z-score）
    mu = roller_mean(L, slow, 1, 'rolling')
    sigma = roller_std(L, slow, 1, 'rolling')
    Z = safe_div(L - mu, sigma).mask(sigma == 0, 0)

    # 5. 价格动量方向
    close_shifted = close.shift(weriod)  # 前 weriod 个 bar 的收盘价
    M = safe_div(close - close_shifted, close_shifted)  # 动量幅度
    D = M.apply(np.sign)  # 方向信号：+1, -1, 0

    # 6. 量价协同合成
    F_raw = Z * D

    # 7. 指数平滑（EWM），使用 alpha 转换为 span
    span = int(2 / alpha - 1) if alpha > 0 else 1
    F_smooth = roller_mean(F_raw, span, 1, 'ewm')  # 内部指数平滑

    # 8. 最终平滑（框架强制）
    alpha_out = roller_mean(F_smooth, window, 1, method)

    return alpha_out
"""
ybf001.py — bar_direction_flow_bayes_001 Core 引擎
bar_direction_flow_bayes_001 因子实现
"""
import pdb
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *


def ybf001(open, high, low, close, volume, openint, window, ewm=False,
           w0=1.0, w1=1.0, w2=1.0, alpha=1.0, beta=0.5, oi_clip=True,
           volume_min=0):
    method = 'ewm' if ewm else 'rolling'

    # Step 1: OHLC 方向代理
    hl_range = high - low
    p1 = safe_div(close - low, hl_range)
    p1 = p1.where(hl_range != 0, 0.5)
    p1 = p1.clip(0, 1)

    p2 = 0.5 + safe_div(close - open, 2 * hl_range)
    p2 = p2.where(hl_range != 0, 0.5)
    p2 = p2.clip(0, 1)

    # Step 2: 贝叶斯后验融合
    theta = (w0 * 0.5 + w1 * p1 + w2 * p2) / (w0 + w1 + w2)
    theta = theta.clip(0, 1)

    # Step 3: 持仓量变化确认项
    d_oi = openint - openint.shift(1)
    if oi_clip:
        d_oi = d_oi.clip(lower=-volume, upper=volume)
    r = safe_div(d_oi, volume)
    confirm = alpha + beta * r

    # Step 4: 最终因子值
    factor_raw = (2 * theta - 1) * confirm * np.sqrt(volume.clip(lower=0))
    factor_raw = factor_raw.where(volume > 0, 0)
    if volume_min > 0:
        factor_raw = factor_raw.where(volume >= volume_min, 0)

    # 最终平滑
    alpha_out = roller_mean(factor_raw, window, 1, method)
    return alpha_out
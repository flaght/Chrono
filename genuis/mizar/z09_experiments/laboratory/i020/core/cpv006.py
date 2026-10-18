"""
cpv006.py — Core 因子计算函数
从说明书 cpv005_resonance_atr_001 演进，参数化实现共振 ATR 调节因子
"""
import pdb
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *


def cpv006(close,
           high,
           low,
           volume,
           openint,
           window,
           werid,
           N1=20,
           N2=20,
           N3=20,
           N4=20,
           n_atr=14,
           gamma=0.5,
           alpha_coef=0.5,
           ewm=False):
    """
    共振 ATR 调节因子：
    pv_z * (1 + gamma * res_norm) * weight
    """
    method = 'ewm' if ewm else 'rolling'

    # ---- Step 1: 原始 PV 信号 ----
    ret_p = safe_div(close - close.shift(1), close.shift(1))  # 价格普通收益率
    ret_v = safe_div(volume - volume.shift(1), volume.shift(1))  # 成交量变化率
    pv_raw = ret_p * ret_v

    # ---- Step 2: PV 信号滚动 z-score ----
    pv_mean = roller_mean(pv_raw, N1, 1, 'rolling')
    pv_std = roller_std(pv_raw, N1, 1, 'rolling')
    pv_z = safe_div(pv_raw - pv_mean, pv_std + 1e-10)

    # ---- Step 3: 价格动量（EMA 斜率）----
    ema_close = roller_mean(close, 5, 1, 'ewm')  # 5 周期 EMA
    mom = ema_close - ema_close.shift(1)

    # ---- Step 4: 动量滚动 z-score ----
    mom_mean = roller_mean(mom, N2, 1, 'rolling')
    mom_std = roller_std(mom, N2, 1, 'rolling')
    mom_z = safe_div(mom - mom_mean, mom_std + 1e-10)

    # ---- Step 5: 持仓量变化率滚动 z-score ----
    oi_ret = safe_div(openint - openint.shift(1), openint.shift(1))
    oi_mean = roller_mean(oi_ret, N3, 1, 'rolling')
    oi_std = roller_std(oi_ret, N3, 1, 'rolling')
    oi_z = safe_div(oi_ret - oi_mean, oi_std + 1e-10)

    # ---- Step 6: 共振因子及归一化 ----
    res_raw = oi_z * mom_z
    res_mean = roller_mean(res_raw, N4, 1, 'rolling')
    res_std = roller_std(res_raw, N4, 1, 'rolling')
    res_norm = safe_div(res_raw - res_mean, res_std + 1e-10)

    # ---- Step 7: ATR 及滚动分位数 ----
    tr = np.maximum(
        high - low,
        np.maximum((high - close.shift(1)).abs(),
                   (low - close.shift(1)).abs()))
    atr = roller_mean(tr, n_atr, 1, 'rolling')
    atr_rank = roller_rank(atr, werid * 20, 1, 'rolling', pct=False)  # 排名 1..w
    atr_percentile = (atr_rank - 1) / (werid * 20 - 1)  # 归一化到 [0,1]
    
    # ---- Step 8: 波动率自适应调节权重 ----
    weight = 1 - alpha_coef * (atr_percentile - 0.5)

    # ---- Step 9: 合成最终因子 ----
    factor_raw = pv_z * (1 + gamma * res_norm) * weight

    # ---- Step 10: 强制最终平滑（框架硬性规范）----
    alpha = roller_mean(factor_raw, window, 1, method)
    return alpha

import numpy as np
import pandas as pd
from lumina.impulse.fixed import *
import pdb
def pareto005(openint, close, volume, window, fast, slow, ewm=False):
    method = 'ewm' if ewm else 'rolling'
    # 硬编码参数（说明书默认值）
    q_low = 0.1
    q_high = 0.9
    epsilon = 1e-6
    k = 5
    w = 5
    lambda_1 = 1/3
    lambda_2 = 1/3
    lambda_3 = 1/3
    alpha_ema = 0.2
    span_ema = int(2 / alpha_ema - 1)  # 9
    
    # ── 步骤 1：持仓量分位数 ──
    Q_low = roller_quantile(openint, q_low, fast, 1, 'rolling')
    Q_high = roller_quantile(openint, q_high, fast, 1, 'rolling')

    # ── 步骤 2：对数比（一步到位） ──
    L = safe_log(Q_high, Q_low.clip(lower=epsilon))

    # ── 步骤 3：滚动标准化 ──
    mu_L = roller_mean(L, slow, 1, method)
    sigma_L = roller_std(L, slow, 1, method)
    Z_oi = safe_div(L - mu_L, sigma_L).fillna(0)  # sigma=0 时补 0

    # ── 步骤 4：动量方向信号 ──
    close_shift_k = close.shift(k)
    M = safe_div(close - close_shift_k, close_shift_k)
    D = np.sign(M)  # -1, 0, 1

    # ── 步骤 5：基础方向性因子 ──
    F_base = Z_oi * D

    # ── 步骤 6：候选特征 ──
    # 成交量变化率
    vol_shift = volume.shift(1)
    V_raw = safe_div(volume - vol_shift, vol_shift.clip(lower=epsilon))
    # 持仓量变化率
    oi_shift = openint.shift(1)
    OI_raw = safe_div(openint - oi_shift, oi_shift.clip(lower=epsilon))
    # 已实现波动率（简单收益率）
    close_shift1 = close.shift(1)
    rets = safe_div(close - close_shift1, close_shift1)
    RV_raw = roller_std(rets, w, 1, 'rolling')

    # 标准化辅助函数
    def standardize(x):
        mu_x = roller_mean(x, slow, 1, method)
        sigma_x = roller_std(x, slow, 1, method)
        return safe_div(x - mu_x, sigma_x).fillna(0)

    V_std = standardize(V_raw)
    OI_std = standardize(OI_raw)
    RV_std = standardize(RV_raw)

    # ── 步骤 7：正交化（滚动回归提取残差） ──
    def orthogonalize(X):
        cov = roller_cov(X, F_base, slow, 1, 'rolling')
        var = roller_var(F_base, slow, 1, 'rolling')
        beta = safe_div(cov, var).fillna(0)  # var=0 时 beta=0
        alpha_reg = roller_mean(X, slow, 1, method) - beta * roller_mean(F_base, slow, 1, method)
        e = X - (alpha_reg + beta * F_base)
        return e

    e1 = orthogonalize(V_std)
    e2 = orthogonalize(OI_std)
    e3 = orthogonalize(RV_std)

    # ── 步骤 8：组合 ──
    F_raw = F_base + lambda_1 * e1 + lambda_2 * e2 + lambda_3 * e3

    # ── 步骤 9：EMA 平滑（固定 span=9） ──
    F_smooth = roller_mean(F_raw, span_ema, 1, 'ewm')

    # ── 强制最终平滑（框架硬性要求） ──
    alpha = roller_mean(F_smooth, window, 1, method)
    return alpha
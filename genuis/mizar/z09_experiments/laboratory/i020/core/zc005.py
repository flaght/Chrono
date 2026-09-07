"""
Core: zc005
"""
import pdb
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *


def zc005(close, volume, amount, openint, window, fast, slow, weriod, ewm=False,
          vcv_window_days=10, ofi_lookback_days=20, rank_window_days=120,
          zscore_lookback_days=120, ic_window_days=120, icir_lookback_days=60,
          forward_return_h_days=5, orthogonal_window_days=120,
          clip_value=3.0, min_obs_ratio=0.8, roll_jump_threshold=20.0,
          eps=1e-12):

    method = 'ewm' if ewm else 'rolling'

    def _zscore(x, lookback):
        mu = roller_mean(x.shift(1), lookback, 1, method)
        sd = roller_std(x.shift(1), lookback, 1, method)
        z = safe_div(x - mu, sd)
        z = z.mask(sd < eps, 0.0)
        return z.clip(-clip_value, clip_value)

    def _icir_weight(z, forward_ret, h, ic_win, icir_win):
        z_lag = z.shift(h)
        ic = roller_corr(z_lag, forward_ret, ic_win, 1, method)
        ic_mean = roller_mean(ic, icir_win, 1, method)
        ic_std = roller_std(ic, icir_win, 1, method)
        return safe_div(ic_mean, ic_std + eps)

    # ---- 日度 OI 变化与换月/异常跳变处理 ----
    raw_dOI = openint - openint.shift(1)
    abs_dOI = raw_dOI.abs()
    jump_base = roller_median(abs_dOI, fast, 1, 'rolling') * roll_jump_threshold
    jump_flag = (abs_dOI > jump_base) & jump_base.notna()

    dOI = raw_dOI.where(~jump_flag)
    dOI_ofi = raw_dOI.where(~jump_flag, 0.0)

    # ---- F1: 量仓同步滚动相关 ----
    F1 = roller_corr(amount, dOI, fast, 1, method)
    valid_pairs = roller_sum(
        (amount.notna() & dOI.notna()).astype(float), fast, 1, 'rolling'
    )
    F1 = F1.where(valid_pairs >= fast * min_obs_ratio, 0.0)

    # ---- F2: 量仓分位背离强度 ----
    VCV = safe_div(
        roller_std(amount, fast, 1, method),
        roller_mean(amount, fast, 1, method)
    )
    oi_activity = roller_mean(dOI.abs(), fast, 1, method)

    pV = roller_rank(VCV, weriod, 1, 'rolling', pct=True)
    pO = roller_rank(oi_activity, weriod, 1, 'rolling', pct=True)
    F2 = pV - pO

    # ---- F3: OFI 滚动标准差与偏度 ----
    price_change = close - close.shift(1)
    sign_price = np.sign(price_change)
    signed_oi = sign_price * dOI_ofi
    ofi = safe_div(signed_oi, volume)

    F3_std = roller_std(ofi, slow, 1, method)
    F3_skew = roller_skew(ofi, slow, 1, 'rolling')
    F3_skew = F3_skew.where(F3_std > eps, 0.0)

    # ---- 特征滚动标准化 ----
    z1 = _zscore(F1, zscore_lookback_days)
    z2 = _zscore(F2, zscore_lookback_days)
    z3 = _zscore(F3_std, zscore_lookback_days)
    z4 = _zscore(F3_skew, zscore_lookback_days)

    # ---- 滚动 ICIR 加权合成 ----
    forward_ret = safe_log(close, close.shift(forward_return_h_days))

    w1 = _icir_weight(z1, forward_ret, forward_return_h_days,
                      ic_window_days, icir_lookback_days)
    w2 = _icir_weight(z2, forward_ret, forward_return_h_days,
                      ic_window_days, icir_lookback_days)
    w3 = _icir_weight(z3, forward_ret, forward_return_h_days,
                      ic_window_days, icir_lookback_days)
    w4 = _icir_weight(z4, forward_ret, forward_return_h_days,
                      ic_window_days, icir_lookback_days)

    weighted_sum = w1 * z1 + w2 * z2 + w3 * z3 + w4 * z4
    weight_abs_sum = w1.abs() + w2.abs() + w3.abs() + w4.abs()
    raw = safe_div(weighted_sum, weight_abs_sum + eps)

    # ---- 已实现波动率 ----
    ret = safe_shift(close, 1)
    rv = np.sqrt(roller_sum(ret * ret, fast, 1, method))

    # ---- 对已实现波动率滚动正交化 ----
    lag_raw = raw.shift(1)
    lag_rv = rv.shift(1)

    beta = safe_div(
        roller_cov(lag_raw, lag_rv, orthogonal_window_days, 1, method),
        roller_var(lag_rv, orthogonal_window_days, 1, method)
    )
    alpha_reg = roller_mean(lag_raw, orthogonal_window_days, 1, method) \
        - beta * roller_mean(lag_rv, orthogonal_window_days, 1, method)

    resid = raw - (alpha_reg + beta * rv)
    resid = resid.where(resid.notna(), raw)

    # ---- 最终标准化 ----
    final_factor = _zscore(resid, zscore_lookback_days)

    # ---- 强制最终平滑 ----
    alpha = roller_mean(final_factor, window, 1, method)
    return alpha
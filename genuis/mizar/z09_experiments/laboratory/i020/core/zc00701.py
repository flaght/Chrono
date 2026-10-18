"""
zc00701_core.py
因子：tail_asym_pv_div_orthogonal_001
实现：lumina.impulse fixed 算子白名单下的 O3 正交尾部量价背离因子
"""

import pdb
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def zc00701(close, high, low, volume, value, openint, window, fast, slow, weriod,
            vol_est_window=60, final_z_window=252, winsor_limit=3.0,
            orth_resid_clip=3.0, eps=1e-12, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    def _robust_z(x, span, clip=3.0):
        med = roller_median(x, span, 1, 'rolling')
        mad = roller_median((x - med).abs(), span, 1, 'rolling')
        z = safe_div(x - med, 1.4826 * mad + eps)
        return z.clip(-clip, clip)

    def _resid_ols(y, x, span):
        beta = safe_div(roller_cov(y, x, span, 1, method),
                        roller_var(x, span, 1, method) + eps)
        return y - beta * x

    # 基础日内量价因子 x_d 的宽表近似
    rets = safe_shift(close, 1)
    price_range = high - low + eps
    pos = safe_div(close - low, price_range)

    low_vol = volume * (1.0 - pos)
    high_vol = volume * pos
    x_vol = safe_log(1.0 + low_vol, 1.0 + high_vol)

    # 状态变量：已实现波动率、趋势强度、持仓量变化
    rv = 10000.0 * roller_std(rets, fast, 1, method)
    vol_est = roller_std(rets, vol_est_window, 1, method)
    trend_num = safe_shift(close, slow)
    trend = safe_div(trend_num, np.sqrt(slow) * vol_est + eps)

    oi_used = openint.where(openint > 0, value)
    oi_num = oi_used - oi_used.shift(slow)
    oi_den = 0.5 * (oi_used + oi_used.shift(slow)) + eps
    oi_chg = safe_div(oi_num, oi_den)

    # 状态变量鲁棒标准化
    v_rv = _robust_z(rv, weriod, winsor_limit)
    v_trend = _robust_z(trend, weriod, winsor_limit)
    v_oi = _robust_z(oi_chg, weriod, winsor_limit)

    # 基准因子 F 的滚动鲁棒标准化代理
    F = _robust_z(x_vol, final_z_window, winsor_limit)

    # 条件尾部不对称度 TA_raw
    q10 = roller_quantile(x_vol, 0.10, weriod, 1, 'rolling')
    q50 = roller_quantile(x_vol, 0.50, weriod, 1, 'rolling')
    q90 = roller_quantile(x_vol, 0.90, weriod, 1, 'rolling')
    ta_raw = safe_div((q90 - q50) - (q50 - q10),
                      q90 - q10 + eps)

    # 稳健 RV 归一化量价背离 PV_raw
    xs = _robust_z(x_vol, weriod, winsor_limit)
    rvs = _robust_z(self_log(rv + eps), weriod, winsor_limit)
    pv_raw = xs - rvs

    # 对基准因子及状态变量滚动残差化，获得正交增量信息
    e_ta = _resid_ols(ta_raw, F, weriod)
    e_ta = _resid_ols(e_ta, v_rv, weriod)
    e_ta = _resid_ols(e_ta, v_trend, weriod)
    e_ta = _resid_ols(e_ta, v_oi, weriod)

    e_pv = _resid_ols(pv_raw, F, weriod)
    e_pv = _resid_ols(e_pv, v_rv, weriod)
    e_pv = _resid_ols(e_pv, v_trend, weriod)
    e_pv = _resid_ols(e_pv, v_oi, weriod)

    # 两个正交增量特征之间再做一次滚动正交化
    beta_orth = safe_div(roller_cov(e_ta, e_pv, weriod, 1, method),
                         roller_var(e_ta, weriod, 1, method) + eps)
    r_pv = e_pv - beta_orth * e_ta
    r_ta = e_ta

    # 正交残差标准化、截断、合成
    n_ta = _robust_z(r_ta, final_z_window, orth_resid_clip)
    n_pv = _robust_z(r_pv, final_z_window, orth_resid_clip)
    raw_comb = (n_ta + n_pv) / 2.0

    f_tail = _robust_z(raw_comb, final_z_window, winsor_limit)

    # 框架强制最终平滑
    alpha = roller_mean(f_tail, window, 1, method)
    return alpha
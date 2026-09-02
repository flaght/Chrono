# -*- encoding:utf-8 -*-
import pdb
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def _zscore_feature(x, lookback, min_obs, eps=1e-12):
    """滚动 z-score，均值/标准差窗口不包含当前值。"""
    x_prev = x.shift(1)
    mu = roller_mean(x_prev, lookback, min_obs, 'rolling')
    sd = roller_std(x_prev, lookback, min_obs, 'rolling')

    mu_valid = mu.notna()
    sd_valid = sd.notna()

    z = safe_div(x - mu, sd)
    z = z.where(mu_valid & sd_valid & sd.gt(eps), 0.0)
    z = z.where(mu_valid & sd_valid)

    return z.clip(-3.0, 3.0)


def _state_icir(feature, ret_fwd, qf1, qf2, rank_window, ic_window,
                icir_window, min_state_obs=20, min_obs=8, low=0.30,
                high=0.70, eps=1e-12):
    """F1/F2 分位状态 ICIR 加权，缺失时回退全局 ICIR。"""
    feat_rank = roller_rank(feature, rank_window, min_obs, 'rolling', pct=True)
    ret_rank = roller_rank(ret_fwd, rank_window, min_obs, 'rolling', pct=True)

    high1 = qf1 >= high
    low1 = qf1 <= low
    mid1 = qf1.notna() & ~high1 & ~low1

    high2 = qf2 >= high
    low2 = qf2 <= low
    mid2 = qf2.notna() & ~high2 & ~low2

    state_hh = high1 & high2
    state_hm = high1 & mid2
    state_hl = high1 & low2
    state_mh = mid1 & high2
    state_mm = mid1 & mid2
    state_ml = mid1 & low2
    state_lh = low1 & high2
    state_lm = low1 & mid2
    state_ll = low1 & low2
    state_valid = qf1.notna() & qf2.notna()

    ic_hh = roller_corr(feat_rank.where(state_hh), ret_rank.where(state_hh),
                        ic_window, min_state_obs, 'rolling')
    ic_hm = roller_corr(feat_rank.where(state_hm), ret_rank.where(state_hm),
                        ic_window, min_state_obs, 'rolling')
    ic_hl = roller_corr(feat_rank.where(state_hl), ret_rank.where(state_hl),
                        ic_window, min_state_obs, 'rolling')
    ic_mh = roller_corr(feat_rank.where(state_mh), ret_rank.where(state_mh),
                        ic_window, min_state_obs, 'rolling')
    ic_mm = roller_corr(feat_rank.where(state_mm), ret_rank.where(state_mm),
                        ic_window, min_state_obs, 'rolling')
    ic_ml = roller_corr(feat_rank.where(state_ml), ret_rank.where(state_ml),
                        ic_window, min_state_obs, 'rolling')
    ic_lh = roller_corr(feat_rank.where(state_lh), ret_rank.where(state_lh),
                        ic_window, min_state_obs, 'rolling')
    ic_lm = roller_corr(feat_rank.where(state_lm), ret_rank.where(state_lm),
                        ic_window, min_state_obs, 'rolling')
    ic_ll = roller_corr(feat_rank.where(state_ll), ret_rank.where(state_ll),
                        ic_window, min_state_obs, 'rolling')

    icir_hh = safe_div(roller_mean(ic_hh, icir_window, 1, 'rolling'),
                       roller_std(ic_hh, icir_window, 1, 'rolling') + eps)
    icir_hm = safe_div(roller_mean(ic_hm, icir_window, 1, 'rolling'),
                       roller_std(ic_hm, icir_window, 1, 'rolling') + eps)
    icir_hl = safe_div(roller_mean(ic_hl, icir_window, 1, 'rolling'),
                       roller_std(ic_hl, icir_window, 1, 'rolling') + eps)
    icir_mh = safe_div(roller_mean(ic_mh, icir_window, 1, 'rolling'),
                       roller_std(ic_mh, icir_window, 1, 'rolling') + eps)
    icir_mm = safe_div(roller_mean(ic_mm, icir_window, 1, 'rolling'),
                       roller_std(ic_mm, icir_window, 1, 'rolling') + eps)
    icir_ml = safe_div(roller_mean(ic_ml, icir_window, 1, 'rolling'),
                       roller_std(ic_ml, icir_window, 1, 'rolling') + eps)
    icir_lh = safe_div(roller_mean(ic_lh, icir_window, 1, 'rolling'),
                       roller_std(ic_lh, icir_window, 1, 'rolling') + eps)
    icir_lm = safe_div(roller_mean(ic_lm, icir_window, 1, 'rolling'),
                       roller_std(ic_lm, icir_window, 1, 'rolling') + eps)
    icir_ll = safe_div(roller_mean(ic_ll, icir_window, 1, 'rolling'),
                       roller_std(ic_ll, icir_window, 1, 'rolling') + eps)

    icir_state = (
        icir_hh.where(state_hh, 0.0)
        + icir_hm.where(state_hm, 0.0)
        + icir_hl.where(state_hl, 0.0)
        + icir_mh.where(state_mh, 0.0)
        + icir_mm.where(state_mm, 0.0)
        + icir_ml.where(state_ml, 0.0)
        + icir_lh.where(state_lh, 0.0)
        + icir_lm.where(state_lm, 0.0)
        + icir_ll.where(state_ll, 0.0)
    )

    ic_global = roller_corr(feat_rank, ret_rank, ic_window, min_state_obs,
                            'rolling')
    icir_global = safe_div(roller_mean(ic_global, icir_window, 1, 'rolling'),
                           roller_std(ic_global, icir_window, 1, 'rolling') + eps)

    icir = icir_state.where(state_valid)
    icir = icir.where(icir.notna(), icir_global)
    icir = icir.where(icir.notna(), 0.0)

    return icir


def zc00501(close, volume, value, openint, window, fast, slow, weriod,
            ewm=False,
            vcv_window_days=10,
            ofi_lookback_days=20,
            rank_window_days=120,
            zscore_lookback_days=120,
            ic_window_days=120,
            icir_lookback_days=60,
            forward_return_h_days=5,
            orthogonal_window_days=120,
            roll_jump_threshold=20.0,
            tail_quantile=0.10,
            state_quantile_low=0.30,
            state_quantile_high=0.70,
            min_state_obs=20,
            min_bar_count=10,
            min_obs_ratio=0.8,
            eps=1e-12):
    method = 'ewm' if ewm else 'rolling'

    vcv_lookback = int(fast) if fast and fast > 0 else vcv_window_days
    rank_lookback = int(slow) if slow and slow > 0 else rank_window_days
    icir_lookback = int(weriod) if weriod and weriod > 0 else icir_lookback_days
    ofi_lookback = max(vcv_lookback, int(ofi_lookback_days))
    zscore_lookback = int(zscore_lookback_days)
    orthogonal_window = int(orthogonal_window_days)

    min_rank_obs = max(10, int(min_obs_ratio * rank_lookback))
    min_day_obs = max(5, int(min_obs_ratio * vcv_lookback))
    min_z_obs = max(10, int(min_obs_ratio * zscore_lookback))
    orthogonal_min_n = 60
    
    rank_lookback = max(rank_lookback, min_rank_obs)
    vcv_lookback = max(max(vcv_lookback, 8),min_day_obs)
    
    # ---------- 换月/异常 OI 跳变处理 ----------
    d_oi_raw = openint - openint.shift(1)
    roll_median_abs_doi = roller_median(d_oi_raw.abs(), rank_lookback,
                                        min_rank_obs, 'rolling')
    roll_jump = d_oi_raw.abs() > roll_jump_threshold * roll_median_abs_doi
    d_oi = d_oi_raw.where(~roll_jump)

    valid_volume = volume > 0

    # bar 级 signed_OI 的日频代理
    bar_move = safe_shift(close, 1)
    signed_oi = np.sign(bar_move) * d_oi
    signed_oi = signed_oi.where(~roll_jump, 0.0)

    # ---------- F1：量仓同步 Spearman 相关 ----------
    rank_value = roller_rank(value, rank_lookback, min_rank_obs, 'rolling',
                             pct=True)
    rank_doi = roller_rank(d_oi, rank_lookback, min_rank_obs, 'rolling',
                           pct=True)
    f1 = roller_corr(rank_value, rank_doi, vcv_lookback, 8, 'rolling')
    f1 = f1.where(f1.notna(), 0.0)

    # ---------- F2：成交额 VCV 与 OI 活动度分位背离 ----------
    vcv = safe_div(roller_std(value, vcv_lookback, min_day_obs, method),
                   roller_mean(value, vcv_lookback, min_day_obs, method))
    oi_activity = roller_mean(d_oi.abs(), vcv_lookback, min_day_obs, method)

    p_v = roller_rank(vcv, rank_lookback, min_rank_obs, 'rolling', pct=True)
    p_o = roller_rank(oi_activity, rank_lookback, min_rank_obs, 'rolling',
                      pct=True)

    f2 = p_v - p_o

    q_f1 = roller_rank(f1, rank_lookback, min_rank_obs, 'rolling', pct=True)
    q_f2 = roller_rank(f2, rank_lookback, min_rank_obs, 'rolling', pct=True)

    # ---------- F3：OFI 滚动标准差与偏度 ----------
    ofi = safe_div(signed_oi, volume.where(valid_volume))
    f3_std = roller_std(ofi, ofi_lookback, min_day_obs, method)
    f3_skew = roller_skew(ofi, ofi_lookback, min_day_obs, 'rolling')
    f3_skew = f3_skew.where(f3_std.notna() & f3_std.gt(eps), 0.0)
    f3_skew = f3_skew.where(f3_std.notna())

    # ---------- F4：极端强度 tail_OFI ----------
    intensity_raw = safe_div(d_oi.abs(), volume.where(valid_volume))
    intensity = intensity_raw.where(~roll_jump)

    intensity_rank = roller_rank(intensity, rank_lookback, min_rank_obs,
                                 'rolling', pct=True)
    tail_mask = intensity_rank.ge(1.0 - tail_quantile) & intensity_rank.notna()
    rest_mask = intensity_rank.notna() & ~tail_mask

    tail_mean = roller_mean(signed_oi.where(tail_mask), rank_lookback,
                            min_bar_count, 'rolling')
    rest_mean = roller_mean(signed_oi.where(rest_mask), rank_lookback,
                            min_bar_count, 'rolling')
    f4 = tail_mean - rest_mean

    # ---------- F5 / F6：分位组合符号交互项 ----------
    f5 = f4 * np.sign(f1) * (q_f2 - 0.5)
    f6 = f4 * np.sign(f2) * (q_f1 - 0.5)

    # ---------- 特征滚动标准化 ----------
    z1 = _zscore_feature(f1, zscore_lookback, min_z_obs)
    z2 = _zscore_feature(f2, zscore_lookback, min_z_obs)
    z3 = _zscore_feature(f3_std, zscore_lookback, min_z_obs)
    z4 = _zscore_feature(f3_skew, zscore_lookback, min_z_obs)
    z5 = _zscore_feature(f4, zscore_lookback, min_z_obs)
    z6 = _zscore_feature(f5, zscore_lookback, min_z_obs)
    z7 = _zscore_feature(f6, zscore_lookback, min_z_obs)

    # ---------- 分状态 ICIR 加权 ----------
    ret_fwd = safe_log(close.shift(-forward_return_h_days), close)

    w1 = _state_icir(z1, ret_fwd, q_f1, q_f2, rank_lookback, ic_window_days,
                     icir_lookback, min_state_obs, min_rank_obs,
                     state_quantile_low, state_quantile_high)
    w2 = _state_icir(z2, ret_fwd, q_f1, q_f2, rank_lookback, ic_window_days,
                     icir_lookback, min_state_obs, min_rank_obs,
                     state_quantile_low, state_quantile_high)
    w3 = _state_icir(z3, ret_fwd, q_f1, q_f2, rank_lookback, ic_window_days,
                     icir_lookback, min_state_obs, min_rank_obs,
                     state_quantile_low, state_quantile_high)
    w4 = _state_icir(z4, ret_fwd, q_f1, q_f2, rank_lookback, ic_window_days,
                     icir_lookback, min_state_obs, min_rank_obs,
                     state_quantile_low, state_quantile_high)
    w5 = _state_icir(z5, ret_fwd, q_f1, q_f2, rank_lookback, ic_window_days,
                     icir_lookback, min_state_obs, min_rank_obs,
                     state_quantile_low, state_quantile_high)
    w6 = _state_icir(z6, ret_fwd, q_f1, q_f2, rank_lookback, ic_window_days,
                     icir_lookback, min_state_obs, min_rank_obs,
                     state_quantile_low, state_quantile_high)
    w7 = _state_icir(z7, ret_fwd, q_f1, q_f2, rank_lookback, ic_window_days,
                     icir_lookback, min_state_obs, min_rank_obs,
                     state_quantile_low, state_quantile_high)

    weighted_sum = (
        w1 * z1 + w2 * z2 + w3 * z3 + w4 * z4
        + w5 * z5 + w6 * z6 + w7 * z7
    )
    weight_abs_sum = (
        w1.abs() + w2.abs() + w3.abs() + w4.abs()
        + w5.abs() + w6.abs() + w7.abs()
    )

    raw = safe_div(weighted_sum, weight_abs_sum)
    raw = raw.where(weight_abs_sum > eps, 0.0)

    # ---------- 与已实现波动率正交化 ----------
    ret_log = safe_shift(close, 1)
    rv = roller_sum(ret_log.pow(2), vcv_lookback, min_day_obs, 'rolling').pow(0.5)

    cov_raw_rv = roller_cov(raw, rv, orthogonal_window, orthogonal_min_n,
                            'rolling')
    var_rv = roller_var(rv, orthogonal_window, orthogonal_min_n, 'rolling')
    beta = safe_div(cov_raw_rv, var_rv)

    rv_mean = roller_mean(rv, orthogonal_window, orthogonal_min_n, 'rolling')
    raw_mean = roller_mean(raw, orthogonal_window, orthogonal_min_n, 'rolling')
    alpha_rv = raw_mean - beta * rv_mean

    resid = raw - (alpha_rv + beta * rv)
    resid = resid.where(beta.notna() & alpha_rv.notna(), raw)

    # ---------- 最终标准化 + 强制最终平滑 ----------
    final_factor = _zscore_feature(resid, zscore_lookback, min_z_obs)
    alpha = roller_mean(final_factor, int(window), 1, method)
    return alpha
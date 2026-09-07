# -*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def zc004(close, volume, window, fast, slow, weriod,
          n_bins_B=5, pseudo_count_alpha=0.5, min_minutes_per_day=60,
          min_valid_days=15, ar_order=1, cv_weight=0.5,
          ar_resid_weight=0.5, min_reg_days=60, eps=1e-12,
          include_overnight_gap=False, ewm=False):
    """
    JUVP-TS 量价联合分布熵与日间稳定性因子（单品种时序版）。

    所有说明书中日频统计统一转换为 lumina.impulse 宽表滚动算子链；
    日窗长度通过 min_minutes_per_day 换算为 bar 数窗。
    """
    method = 'ewm' if ewm else 'rolling'
    d_bars = max(int(min_minutes_per_day), 1)

    # Step 1: 数据清洗与分钟绝对收益
    ret = safe_shift(close, 1).abs()

    # 默认剔除隔夜/跨节跳空
    if not include_overnight_gap:
        day = pd.Series(close.index.normalize(), index=close.index)
        is_first = day != day.shift(1)
        ret = ret.where(~is_first, np.nan)

    valid = (close > 0) & (volume > 0) & ret.notna()
    ret = ret.where(valid)
    vol = volume.where(valid)

    # 日窗 -> bar 数窗
    roll_bars = max(int(weriod) * d_bars, 1)
    min_nmi_periods = max(int(min_valid_days) * d_bars, 1)

    # Step 2: 滚动秩分位分箱（经验 Copula 离散化）
    vol_rank = roller_rank(vol, wspan=roll_bars, min_periods=1, method='rolling', pct=True)
    ret_rank = roller_rank(ret, wspan=roll_bars, min_periods=1, method='rolling', pct=True)

    bin_v = np.ceil(vol_rank * n_bins_B).clip(1, n_bins_B)
    bin_r = np.ceil(ret_rank * n_bins_B).clip(1, n_bins_B)

    denom_bins = max(float(n_bins_B - 1), 1.0)
    joint_agree = 1.0 - (bin_v - bin_r).abs() / denom_bins

    # Step 3: 量价同步强度代理（NMI 的连续滚动近似）
    sync_raw = roller_mean(joint_agree, roll_bars, min_nmi_periods, method)
    # Laplace 风格的轻平滑，使估计更稳定
    sync = (sync_raw * (1.0 + pseudo_count_alpha) + pseudo_count_alpha) / (1.0 + 2.0 * pseudo_count_alpha)

    # 日间均值、标准差、变异系数
    avg_nmi = roller_mean(sync, roll_bars, min_nmi_periods, method)
    std_nmi = roller_std(sync, roll_bars, min_nmi_periods, method)
    cv_sync = safe_div(std_nmi, avg_nmi + eps)

    # AR(1) 残差标准差
    ar_lag = max(int(ar_order), 1)
    lag_sync = sync.shift(ar_lag)
    avg_lag = roller_mean(lag_sync, roll_bars, min_nmi_periods, method)
    cov_lag = roller_cov(sync, lag_sync, roll_bars, min_nmi_periods, method)
    var_lag = roller_var(lag_sync, roll_bars, min_nmi_periods, method)
    beta_ar = safe_div(cov_lag, var_lag + eps)

    ar_resid = sync - (avg_nmi + beta_ar * (lag_sync - avg_lag))
    ar_resid_std = roller_std(ar_resid, roll_bars, min_nmi_periods, method)

    # Step 4: 组合不稳定性
    instability = (
        cv_weight * cv_sync
        + ar_resid_weight * safe_div(ar_resid_std, avg_nmi + eps)
    )

    # Step 5: 滚动标准化与负向稳定性分数
    std_bars = max(int(fast) * d_bars, 1)
    std_periods = max(int(min_valid_days) * d_bars, 1)

    mu_inst = roller_mean(instability, std_bars, std_periods, method)
    sd_inst = roller_std(instability, std_bars, std_periods, method)
    z_inst = safe_div(instability - mu_inst, sd_inst + eps)
    stability = -z_inst

    # Step 6: UTD 期货版
    volume_sum = roller_sum(vol, roll_bars, 1, 'rolling')
    volume_share = safe_div(vol, volume_sum)
    turn_vol_daily = roller_std(volume_share, roll_bars, 1, 'rolling')

    utd_futures = safe_div(
        roller_std(turn_vol_daily, roll_bars, 1, 'rolling'),
        roller_mean(turn_vol_daily, roll_bars, 1, 'rolling') + eps
    )

    # Step 6: 对 UTD 做滚动时序残差化
    resid_bars = max(int(slow) * d_bars, 1)
    reg_periods = max(int(min_reg_days) * d_bars, 1)

    avg_stab = roller_mean(stability, resid_bars, reg_periods, method)
    avg_utd = roller_mean(utd_futures, resid_bars, reg_periods, method)
    cov_stab_utd = roller_cov(stability, utd_futures, resid_bars, reg_periods, method)
    var_utd = roller_var(utd_futures, resid_bars, reg_periods, method)
    beta_utd = safe_div(cov_stab_utd, var_utd + eps)

    pred_stab = avg_stab + beta_utd * (utd_futures - avg_utd)
    resid_stab = stability - pred_stab
    resid_sigma = roller_std(resid_stab, resid_bars, reg_periods, method)

    juvp_raw = safe_div(resid_stab, resid_sigma + eps)

    # 回归样本不足时退化为 stability
    obs_cnt = roller_sum((~resid_stab.isna()).astype(float), resid_bars, 1, 'rolling')
    has_reg = (obs_cnt >= reg_periods) & resid_sigma.notna()
    juvp_ts = juvp_raw.where(has_reg, stability)

    # 框架强制最终平滑
    alpha = roller_mean(juvp_ts, window, 1, method)
    return alpha
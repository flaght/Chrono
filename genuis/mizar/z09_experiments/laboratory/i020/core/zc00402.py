# -*- encoding:utf-8 -*-
"""
core/zc00402.py
JUVP-DIVERGENCE 量仓背离状态的条件增量因子（滚动版）

终局因子说明书 `juvp_divergence_cond_state_001` 的 lumina.impulse 滚动实现：
1. 成交量变化 dv 与持仓量变化 doi 按滚动中位数切成高/低两档，组合成 4 类量仓背离状态；
2. 前向收益标签 F_i = log(close_{i+fast}) - log(close_i)，并因果滞后 fast 根 K 线，
   保证当前时刻只使用已经实现的标签（i + fast <= t）；
3. 在滚动窗口 weriod 内估计各类状态的条件均值/标准差/偏度/分位展宽，
   构造方向分、展宽分、不对称分；
4. 用 slow 窗口滚动标准化并按 w_dir/w_quant/w_skew 加权组合 raw_div；
5. 对基础量价联合代理（volume-abs_ret 滚动相关）与 OI-CMI 代理做因果滚动正交化；
6. 最终用 window 做框架强制平滑。
"""
import pdb
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *


def zc00402(close, volume, openint, window, fast, slow, weriod, ewm=False,
            n_change_bins=3,
            cond_quantiles=(0.1, 0.5, 0.9),
            w_dir=0.5,
            w_quant=0.3,
            w_skew=0.2,
            use_oi_cmi=True,
            eps=1e-12):
    """
    量仓背离状态的条件增量因子。

    Parameters
    ----------
    close : DataFrame
        宽表分钟收盘价。
    volume : DataFrame
        宽表分钟成交量。
    openint : DataFrame
        宽表分钟持仓量（换月/主力切换跳变应已在外部置为缺失）。
    window : int
        框架强制最终平滑窗口。
    fast : int
        前向收益标签周期 K（默认说明书为 300 分钟）。
    slow : int
        滚动标准化 / 正交化回归窗口。
    weriod : int
        条件状态估计滚动窗口。
    ewm : bool
        是否使用 ewm 平滑；分位/协方差等统计仍使用 rolling。
    """
    method = 'ewm' if ewm else 'rolling'
    # roller_quantile 仅支持 rolling
    roll_only = 'rolling'

    # ---------- Step 1: 基础序列 ----------
    # 分钟对数涨跌幅绝对值
    abs_ret = safe_shift(close, 1).abs()
    # 成交量变化
    dv = volume - volume.shift(1)
    # 持仓量变化
    doi = openint - openint.shift(1)

    # 前向收益标签：F_i = log(close_{i+fast}) - log(close_i)
    # safe_shift 带负数 drift 后取负号即为前向对数收益
    fwd_log_ret = -safe_shift(close, -fast)
    # 因果对齐：t 时刻只能看到 t-fast 处已经实现的前向收益标签
    fwd_label = fwd_log_ret.shift(fast)

    # ---------- Step 2: 量仓背离状态分箱 ----------
    # 经验中位阈值（n_change_bins 分位中的中间分位，默认 50%）
    state_thr_q = 0.5
    dv_thr = roller_quantile(dv, state_thr_q, weriod, 1, roll_only)
    doi_thr = roller_quantile(doi, state_thr_q, weriod, 1, roll_only)

    dv_high = dv > dv_thr
    dv_low = dv <= dv_thr
    doi_high = doi > doi_thr
    doi_low = doi <= doi_thr

    G1 = (dv_high & doi_high).astype(float)  # 量增仓增
    G2 = (dv_high & doi_low).astype(float)   # 量增仓减
    G3 = (dv_low & doi_high).astype(float)   # 量减仓增
    G4 = (dv_low & doi_low).astype(float)    # 量减仓减

    # 严格因果对齐：状态发生在 fast 根 K 线之前
    G1s = G1.shift(fast)
    G2s = G2.shift(fast)
    G3s = G3.shift(fast)
    G4s = G4.shift(fast)

    # ---------- Step 3: 条件矩 / 条件分位 ----------
    def cond_moment(mask):
        den = roller_sum(mask, weriod, 1, method)
        m1 = safe_div(roller_sum(fwd_label * mask, weriod, 1, method), den)
        m2 = safe_div(roller_sum(fwd_label * fwd_label * mask, weriod, 1, method), den)
        m3 = safe_div(roller_sum(fwd_label * fwd_label * fwd_label * mask, weriod, 1, method), den)
        var = (m2 - m1 * m1).clip(lower=0)
        sd = np.sqrt(var)
        skew = safe_div(m3 - 3 * m1 * m2 + 2 * m1 * m1 * m1,
                        np.maximum(sd ** 3, eps))
        return m1, sd, skew

    def cond_quantile(mask, tau):
        masked = fwd_label.where(mask > 0, np.nan)
        return roller_quantile(masked, tau, weriod, 1, roll_only)

    mu1, sd1, skew1 = cond_moment(G1s)
    mu2, sd2, skew2 = cond_moment(G2s)
    mu3, sd3, skew3 = cond_moment(G3s)
    mu4, sd4, skew4 = cond_moment(G4s)

    # ---------- Step 4: 方向性背离强度与状态不对称性 ----------
    # 量仓背离方向分：量增仓减 vs 量减仓增 的收益差，减去同步增减状态的差异
    div_score = (mu2 - mu3) - (mu1 - mu4)

    # 条件分位展宽分
    q_low = cond_quantiles[0]
    q_high = cond_quantiles[-1]
    qs1 = cond_quantile(G1s, q_high) - cond_quantile(G1s, q_low)
    qs2 = cond_quantile(G2s, q_high) - cond_quantile(G2s, q_low)
    qs3 = cond_quantile(G3s, q_high) - cond_quantile(G3s, q_low)
    qs4 = cond_quantile(G4s, q_high) - cond_quantile(G4s, q_low)
    q_spread = (qs1 + qs2 + qs3 + qs4) / 4.0

    # 条件偏度不对称分
    max_skew = skew1.where(skew1 >= skew2, skew2)
    max_skew = max_skew.where(max_skew >= skew3, skew3)
    max_skew = max_skew.where(max_skew >= skew4, skew4)
    min_skew = skew1.where(skew1 <= skew2, skew2)
    min_skew = min_skew.where(min_skew <= skew3, skew3)
    min_skew = min_skew.where(min_skew <= skew4, skew4)
    skew_spread = max_skew - min_skew

    # ---------- Step 5: 滚动标准化与加权组合 ----------
    def rolling_z(x, lookback):
        mu_x = roller_mean(x, lookback, 1, method)
        sd_x = roller_std(x, lookback, 2, method)
        return safe_div(x - mu_x, np.maximum(sd_x, eps))

    z_div = rolling_z(div_score, slow)
    z_quant = rolling_z(q_spread, slow)
    z_skew = rolling_z(skew_spread, slow)
    raw_div = w_dir * z_div + w_quant * z_quant + w_skew * z_skew

    # ---------- Step 6: 对基础 JUVP / OI-CMI 代理的因果正交化 ----------
    # 基础 JUVP 代理：量-绝对收益滚动相关
    base_juvp = roller_corr(volume, abs_ret, weriod, 2, roll_only)
    beta_b = safe_div(roller_cov(raw_div, base_juvp, slow, 2, roll_only),
                      np.maximum(roller_var(base_juvp, slow, 2, roll_only), eps))
    resid = raw_div - beta_b * base_juvp

    if use_oi_cmi:
        # OI-CMI 代理：持仓量变化-绝对收益滚动相关
        base_ncmi = roller_corr(doi, abs_ret, weriod, 2, roll_only)
        # 先对 base_juvp 做 Gram-Schmidt 正交化
        beta_nc = safe_div(
            roller_cov(base_ncmi, base_juvp, slow, 2, roll_only),
            np.maximum(roller_var(base_juvp, slow, 2, roll_only), eps))
        base_ncmi_orth = base_ncmi - beta_nc * base_juvp
        beta2 = safe_div(
            roller_cov(resid, base_ncmi_orth, slow, 2, roll_only),
            np.maximum(roller_var(base_ncmi_orth, slow, 2, roll_only), eps))
        resid = resid - beta2 * base_ncmi_orth

    # 残差标准化：模拟 OLS 残差 / sigma_e
    alpha_raw = safe_div(resid,
                         np.maximum(roller_std(resid, slow, 2, roll_only), eps))

    # 框架强制最终平滑
    alpha = roller_mean(alpha_raw, window, 1, method)
    return alpha
# -*- encoding:utf-8 -*-
import pdb
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *


def zc00401(close, volume, openint, window, fast, slow, weriod, ewm=False,
            label_horizon_minutes_K=300,
            signed_z_weight=0.5,
            eps=1e-12):
    """
    JUVP-OI 超前-滞后条件依赖增量因子（滚动代理实现）。

    仅在 white-list 算子内构造：
    1. 基础 JUVP 状态：量/绝对收益滚动等级相关性 + 不稳定性代理；
    2. OI/量变领先滞后状态：dv/doi 与已实现 K 分钟对数收益的滚动相关性；
    3. 对基础 JUVP 做滚动回归正交化，输出标准化残差。
    """
    method = 'ewm' if ewm else 'rolling'

    # ---------- Step 0/1: 基础 JUVP 状态 ----------
    rets = safe_shift(close, 1)
    # 绝对收益，避免使用未登记方法，用 where 实现
    abs_ret = rets.where(rets > 0, -rets)

    # 量价依赖状态（NMI 代理）
    vol_rank = roller_rank(volume, fast, 1, 'rolling')
    ret_rank = roller_rank(abs_ret, fast, 1, 'rolling')
    nmi_proxy = roller_corr(vol_rank, ret_rank, fast, 1, 'rolling')

    # 不稳定性：CV + AR(1) 残差标准差
    nmi_mean = roller_mean(nmi_proxy, slow, 1, 'rolling')
    nmi_std = roller_std(nmi_proxy, slow, 1, 'rolling')
    cv = safe_div(nmi_std, nmi_mean + eps)
    ar_resid = roller_std(nmi_proxy - nmi_proxy.shift(1), slow, 1, 'rolling')
    instability = 0.5 * cv + 0.5 * safe_div(ar_resid, nmi_mean + eps)

    # stability = -z(instability)
    inst_mean = roller_mean(instability, slow, 1, 'rolling')
    inst_std = roller_std(instability, slow, 1, 'rolling')
    stability = -safe_div(instability - inst_mean, inst_std + eps)

    # 期货版 UTD 残差化：与成交量趋势正交
    vol_trend = roller_mean(volume, weriod, 1, 'rolling')
    beta_base = safe_div(
        roller_cov(stability, vol_trend, slow, 1, 'rolling'),
        roller_var(vol_trend, slow, 1, 'rolling') + eps
    )
    juvp_base = (
        stability
        - roller_mean(stability, slow, 1, 'rolling')
        - beta_base * (vol_trend - roller_mean(vol_trend, slow, 1, 'rolling'))
    )

    # ---------- Step 2/3: OI/量变 领先滞后条件状态 ----------
    dv = volume - volume.shift(1)
    doi = openint - openint.shift(1)

    # 当前量/持仓量水平联合状态（条件 Z 代理）
    vol_level_rank = roller_rank(volume, fast, 1, 'rolling')
    oi_level_rank = roller_rank(openint, fast, 1, 'rolling')
    cond_state = (vol_level_rank + oi_level_rank) / 2.0

    # 量变 + 持仓量变化联合特征（特征 X 代理）
    dv_rank = roller_rank(dv, fast, 1, 'rolling')
    doi_rank = roller_rank(doi, fast, 1, 'rolling')
    lead_source = safe_div(dv_rank + doi_rank, cond_state + eps)

    # 未来 K 分钟对数收益，并 shift(K) 对齐为“当前已知标签”，确保无前视
    future_log_ret_raw = safe_log(close.shift(-label_horizon_minutes_K), close)
    future_log_ret_known = future_log_ret_raw.shift(label_horizon_minutes_K)

    # 条件互信息代理 + 方向性 spread
    cmi_proxy = roller_corr(lead_source, future_log_ret_known, fast, 1, 'rolling')
    spread_proxy = roller_corr(doi, future_log_ret_known, fast, 1, 'rolling')

    # 滚动标准化
    cmi_mean = roller_mean(cmi_proxy, slow, 1, 'rolling')
    cmi_std = roller_std(cmi_proxy, slow, 1, 'rolling')
    spread_mean = roller_mean(spread_proxy, slow, 1, 'rolling')
    spread_std = roller_std(spread_proxy, slow, 1, 'rolling')

    z_cmi = safe_div(cmi_proxy - cmi_mean, cmi_std + eps)
    z_spread = safe_div(spread_proxy - spread_mean, spread_std + eps)
    raw_increment = (1 - signed_z_weight) * z_cmi + signed_z_weight * z_spread

    # ---------- Step 4/5: 对基础 JUVP 的因果滚动正交化 ----------
    beta_orth = safe_div(
        roller_cov(raw_increment, juvp_base, slow, 1, 'rolling'),
        roller_var(juvp_base, slow, 1, 'rolling') + eps
    )
    resid = (
        raw_increment
        - roller_mean(raw_increment, slow, 1, 'rolling')
        - beta_orth * (juvp_base - roller_mean(juvp_base, slow, 1, 'rolling'))
    )
    sigma_e = roller_std(resid, slow, 1, 'rolling')
    alpha_raw = safe_div(resid, sigma_e + eps)

    # ---------- 强制最终平滑 ----------
    alpha = roller_mean(alpha_raw, window, 1, method)
    return alpha
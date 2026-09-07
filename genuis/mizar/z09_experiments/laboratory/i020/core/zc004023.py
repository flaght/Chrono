# -*- encoding:utf-8 -*-
import numpy as np
from lumina.impulse.fixed import *


def zc004023(close, volume, openint, window, weriod, q=0.2,
             n_tail_bins=5, w_tail_volume=1.0, w_tail_oi=1.0,
             w_tail_return=1.0, w_asym=0.4, w_wmi=0.3, w_dir=0.3,
             eps=1e-12, ewm=False):
    """
    zc004023: 上下尾 Copula 非对称依赖因子。

    在宽表 DataFrame 上逐列独立计算滚动尾部依赖、加权尾互信息代理
    以及方向性极端条件特征，最终做框架要求的滚动平滑。
    """
    method = 'ewm' if ewm else 'rolling'

    # ---- Step 1: 数据清洗 ----
    close = close.where(close > 0, np.nan)
    volume = volume.where(volume > 0, np.nan)
    openint = openint.where(openint > 0, np.nan)

    ret = safe_shift(close, 1)
    dv = volume - volume.shift(1)
    doi = openint - openint.shift(1)

    # ---- Step 2: 概率积分变换 / 秩均匀化 ----
    # roller_rank 仅支持 rolling，不支持 ewm，因此这里固定使用 'rolling'
    u_dv = roller_rank(dv, weriod, 1, 'rolling', pct=True)
    u_doi = roller_rank(doi, weriod, 1, 'rolling', pct=True)
    u_r = roller_rank(ret, weriod, 1, 'rolling', pct=True)

    # ---- 上下尾事件 ----
    up_v = (u_dv > 1 - q).astype(float)
    lo_v = (u_dv < q).astype(float)
    up_oi = (u_doi > 1 - q).astype(float)
    lo_oi = (u_doi < q).astype(float)
    up_r = (u_r > 1 - q).astype(float)
    lo_r = (u_r < q).astype(float)

    # ---- Step 3/4: 极值象限加权 ----
    bin_dv = np.ceil(u_dv * n_tail_bins)
    bin_doi = np.ceil(u_doi * n_tail_bins)
    bin_r = np.ceil(u_r * n_tail_bins)

    ext_v = ((bin_dv <= 1) | (bin_dv >= n_tail_bins)).astype(float)
    ext_oi = ((bin_doi <= 1) | (bin_doi >= n_tail_bins)).astype(float)
    ext_r = ((bin_r <= 1) | (bin_r >= n_tail_bins)).astype(float)

    w_extreme = (1.0 + w_tail_volume * ext_v + w_tail_oi * ext_oi +
                 w_tail_return * ext_r)

    up_cnt_v = roller_sum(up_v * w_extreme, weriod, 1, method)
    lo_cnt_v = roller_sum(lo_v * w_extreme, weriod, 1, method)
    up_cnt_oi = roller_sum(up_oi * w_extreme, weriod, 1, method)
    lo_cnt_oi = roller_sum(lo_oi * w_extreme, weriod, 1, method)
    up_cnt_r = roller_sum(up_r * w_extreme, weriod, 1, method)
    lo_cnt_r = roller_sum(lo_r * w_extreme, weriod, 1, method)

    # ---- 上下尾 Copula 依赖系数 ----
    asym_vr = (
        safe_div(roller_sum(up_v * up_r * w_extreme, weriod, 1, method), up_cnt_v) -
        safe_div(roller_sum(lo_v * lo_r * w_extreme, weriod, 1, method), lo_cnt_v)
    )

    asym_oi_r = (
        safe_div(roller_sum(up_oi * up_r * w_extreme, weriod, 1, method), up_cnt_oi) -
        safe_div(roller_sum(lo_oi * lo_r * w_extreme, weriod, 1, method), lo_cnt_oi)
    )

    asym_v_oi = (
        safe_div(roller_sum(up_v * up_oi * w_extreme, weriod, 1, method), up_cnt_v) -
        safe_div(roller_sum(lo_v * lo_oi * w_extreme, weriod, 1, method), lo_cnt_v)
    )

    # ---- 极值加权互信息代理 ----
    joint_up_cnt = roller_sum(up_v * up_oi * up_r * w_extreme, weriod, 1, method)
    joint_lo_cnt = roller_sum(lo_v * lo_oi * lo_r * w_extreme, weriod, 1, method)

    indep_up_cnt = (up_cnt_v * up_cnt_oi * up_cnt_r) / (weriod ** 3)
    indep_lo_cnt = (lo_cnt_v * lo_cnt_oi * lo_cnt_r) / (weriod ** 3)

    mi_up = safe_log(joint_up_cnt + 1.0, indep_up_cnt + 1.0)
    mi_lo = safe_log(joint_lo_cnt + 1.0, indep_lo_cnt + 1.0)

    wmi_tail = mi_up + mi_lo

    # ---- 方向性极端条件特征 ----
    # 使用因果滚动尾部依赖中的正负收益不对称部分作为方向性代理
    tail_dir = asym_vr + asym_oi_r

    # ---- 滚动标准化辅助函数 ----
    def _zscore(x, period, method):
        mean_x = roller_mean(x, period, 1, method)
        std_x = roller_std(x, period, 1, method)
        std_safe = std_x.where(std_x > eps, np.nan)
        return safe_div(x - mean_x, std_safe)

    asym_lambda = (asym_vr + asym_oi_r + asym_v_oi) / 3.0

    z_asym = _zscore(asym_lambda, weriod, method)
    z_wmi = _zscore(wmi_tail, weriod, method)
    z_dir = _zscore(tail_dir, weriod, method)

    # ---- 组合尾部特征 ----
    raw_tail = w_asym * z_asym + w_wmi * z_wmi + w_dir * z_dir

    # ---- 框架强制最终平滑 ----
    alpha = roller_mean(raw_tail, window, 1, method)

    return alpha
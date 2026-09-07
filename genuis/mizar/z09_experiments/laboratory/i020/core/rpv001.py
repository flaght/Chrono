"""
rpv001_core.py — RPV 因子保真还原版
定义：RPV = zscore(CCOIV) - zscore(COV)
CCOIV：日内 30 分钟窗口内分钟收益率与成交量的相关系数（反转效应）
COV：错配相关系数（用滞后 30 分钟成交量与当前收益率的相关系数近似隔夜错配）
随后接 RV 归一化(10分钟) → 30分钟滚动z-score → 输出连续alpha
"""
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *


def rpv001(close,
           volume,
           window=1,
           weriod=30,
           rv_window=10,
           min_samples=15,
           ewm=False):
    """
    参数说明：
        close: 分钟收盘价宽表 (DataFrame, index=time, columns=code)
        volume: 分钟成交量宽表 (DataFrame, index=time, columns=code)
        window: 最终平滑窗口（仅用于最后输出平滑）
        weriod: 30分钟滚动窗口（业务核心周期）
        rv_window: 已实现波动率计算窗口（默认10分钟）
        min_samples: 最小有效样本数（默认15）
        ewm: 是否使用指数加权（默认False）
    返回: 连续浮点型 alpha
    """
    method = 'ewm' if ewm else 'rolling'

    # 1. 分钟收益率
    rets = safe_shift(close, 1)

    # 2. CCOIV: 30分钟窗口内, 分钟收益率与成交量的相关系数
    ccoiv = roller_corr(rets, volume, weriod, min_samples, method)

    # 3. COV: 错配相关系数 — 用滞后30分钟成交量与当前收益率的相关系数近似隔夜错配
    volume_lag = safe_shift(volume, weriod)
    cov = roller_corr(volume_lag, rets, weriod, min_samples, method)

    # 4. 标准化 CCOIV 与 COV
    ccoiv_mean = roller_mean(ccoiv, weriod, min_samples, method)
    ccoiv_std = roller_std(ccoiv, weriod, min_samples, method)
    cov_mean = roller_mean(cov, weriod, min_samples, method)
    cov_std = roller_std(cov, weriod, min_samples, method)

    ccoiv_norm = safe_div(ccoiv - ccoiv_mean, ccoiv_std)
    cov_norm = safe_div(cov - cov_mean, cov_std)

    # 5. RPV = 标准化CCOIV - 标准化COV
    rpv = ccoiv_norm - cov_norm

    # 6. 已实现波动率 RV = sqrt( sum(Δlog(P))^2, window=rv_window )
    rv = roller_sum(rets**2, rv_window, min_periods=rv_window, method=method) ** (0.5)

    # 7. RV 归一化: RPV_norm = RPV / RV
    rpv_norm = safe_div(rpv, rv)

    # 8. 30分钟滚动 z-score
    z_mean = roller_mean(rpv_norm, weriod, min_samples, method)
    z_std = roller_std(rpv_norm, weriod, min_samples, method)
    z_score = safe_div(rpv_norm - z_mean, z_std)

    # 9. 最终平滑（window 仅用于此处）
    alpha = roller_mean(z_score, window, 1, method)
    return alpha
import numpy as np
from lumina.impulse.fixed import (
    safe_shift,
    safe_div,
    roller_mean,
    roller_std,
    roller_corr,
    roller_quantile,
)


def pv_oi_adaptive(close, openint, volume, window, weriod, ewm=False):
    """
    O2 波动率自适应增强的修正持仓量价量相关性因子。
    """
    method = "ewm" if ewm else "rolling"

    # 1. 成交量加权修正持仓量
    vol_mean = roller_mean(volume, weriod, 1, method)
    adj_oi = openint * safe_div(volume, vol_mean)

    # 2. PV 相关系数
    pv = roller_corr(close, adj_oi, weriod, 1, method)

    # 3. 已实现波动率 (RV)
    rets = safe_shift(close, 1)
    rv = roller_std(rets, weriod, 1, method)

    # 4. 波动率历史均值（长窗口，使用 weriod 的整数倍）
    long_window = max(weriod * 2, 60)
    rv_mean = roller_mean(rv, long_window, 1, method)

    # 5. 波动率自适应缩放系数（连续衰减形式）
    base_threshold = 0.5
    rv_ratio = safe_div(rv, rv_mean)
    adaptive_scale = np.maximum(0, 1 - base_threshold / rv_ratio)

    # 6. 趋势强度门控（t 统计量近似 ADX，连续单调函数）
    trend = safe_div(
        roller_mean(rets, weriod, 1, method),
        roller_std(rets, weriod, 1, method),
    )
    gate = np.clip(np.abs(trend), 0, 1)  # |trend| 在 [0, 1] 线性，≥1 时全门控

    # 7. 成交量质量检查（连续压缩，低量时段信号衰减）
    vol_quantile = roller_quantile(volume, 0.2, 20, 1, "rolling")
    vol_qual = safe_div(volume, vol_quantile)
    vol_qual = np.minimum(vol_qual, 1.0)  # 限制最大为 1，防止异常放大

    # 8. 合成原始信号
    alpha_raw = pv * adaptive_scale * gate * vol_qual

    # 9. 最终平滑
    alpha = roller_mean(alpha_raw, window, 1, method)
    return alpha

# -*- coding: utf-8 -*-
import pdb
from lumina.impulse.fixed import *


def _robust_z(x, win, winsor_limit=3.0, eps=1e-12):
    med = roller_median(x, win, 1, 'rolling')
    mad = roller_median((x - med).abs(), win, 1, 'rolling')
    z = safe_div(x - med, 1.4826 * mad + eps)
    return z.clip(-winsor_limit, winsor_limit)


def zc007(close, high, low, volume, openint, window, weriod, ewm=False,
          price_band_alpha=0.20,
          vol_lookback=5,
          trend_lookback=10,
          vol_est_window=60,
          oi_lookback=5,
          state_window=250,
          n_states=4,
          cdf_window=500,
          map_window=504,
          final_z_window=252,
          winsor_limit=3.0,
          eps=1e-12):
    method = 'ewm' if ewm else 'rolling'

    # 1. 原始日内量价不平衡因子
    rolling_high = roller_max(close, vol_lookback, 1, 'rolling')
    rolling_low = roller_min(close, vol_lookback, 1, 'rolling')
    price_pos = safe_div(close - rolling_low, rolling_high - rolling_low + eps)

    low_flag = price_pos <= price_band_alpha
    high_flag = price_pos >= 1.0 - price_band_alpha

    low_volume = roller_sum(volume * low_flag, weriod, 1, method)
    high_volume = roller_sum(volume * high_flag, weriod, 1, method)
    raw_x = safe_log(1.0 + low_volume, 1.0 + high_volume)

    # 2. 状态变量与鲁棒标准化
    rets = safe_shift(close, 1)

    rv = 10000.0 * roller_mean(rets * rets, vol_lookback, 1, method).pow(0.5)

    trend_log = safe_log(close, close.shift(trend_lookback))
    vol_scale = roller_std(rets, vol_est_window, 1, method)
    trend = safe_div(trend_log, (trend_lookback ** 0.5) * vol_scale + eps)

    oi_change = openint - openint.shift(oi_lookback)
    oi_level = 0.5 * (openint + openint.shift(oi_lookback)) + eps
    oi = safe_div(oi_change, oi_level)

    v_rv = _robust_z(rv, state_window, winsor_limit, eps)
    v_trend = _robust_z(trend, state_window, winsor_limit, eps)
    v_oi = _robust_z(oi, state_window, winsor_limit, eps)

    # 3. 滚动分位数状态中心 + 逆距离软概率，近似 GMM 责任
    weights = []
    for s in range(n_states):
        q = (s + 1) / (n_states + 1)
        c_rv = roller_quantile(v_rv, q, state_window, 1, 'rolling')
        c_tr = roller_quantile(v_trend, q, state_window, 1, 'rolling')
        c_oi = roller_quantile(v_oi, q, state_window, 1, 'rolling')
        dist = (v_rv - c_rv) ** 2 + (v_trend - c_tr) ** 2 + (v_oi - c_oi) ** 2
        weights.append(1.0 / (1.0 + dist))

    wsum = weights[0]
    for w in weights[1:]:
        wsum = wsum + w

    probs = [safe_div(w, wsum) for w in weights]

    # 4. 条件经验 CDF 对齐
    u_global = roller_rank(raw_x, cdf_window, 1, 'rolling', pct=True)

    u_state = 0.0
    for s in range(n_states):
        q = (s + 1) / (n_states + 1)
        center_s = roller_quantile(raw_x, q, cdf_window, 1, 'rolling')
        u_s = roller_rank(raw_x - center_s, cdf_window, 1, 'rolling', pct=True)
        u_state = u_state + probs[s] * u_s

    u_aligned = u_state

    # 5. 单调映射近似 + 鲁棒标准化
    u_map = roller_mean(u_aligned, map_window, 1, method)
    yhat = u_map - 0.5

    yhat_med = roller_median(yhat, final_z_window, 1, 'rolling')
    yhat_mad = roller_median((yhat - yhat_med).abs(), final_z_window, 1, 'rolling')
    alpha_raw = safe_div(yhat - yhat_med, 1.4826 * yhat_mad + eps)

    # 6. 强制最终平滑
    alpha = roller_mean(alpha_raw, window, 1, method)
    return alpha
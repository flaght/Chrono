# -*- encoding: utf-8 -*-
import numpy as np

from lumina.impulse.fixed import *


def zc006(close, high, low, value, window, weriod, ewm=False,
          beta_buy=0.88,
          beta_sell=0.88,
          lambda_buy=1.0,
          lambda_sell=2.25,
          k_cap=5.0,
          gamma=0.5,
          q_ref_lookback=20,
          q_ref_quantile=0.5,
          min_vol_bp=1.0,
          eps=1e-12):
    method = 'ewm' if ewm else 'rolling'

    # 日度波动率标尺：使用滚动对数收益标准差近似已实现波动率
    log_ret = safe_shift(close, 1)
    daily_log_vol = roller_std(log_ret, weriod, 1, method)
    vol_scale = daily_log_vol * close
    min_vol_scale = close * (min_vol_bp / 10000.0)
    vol_scale = vol_scale.where(vol_scale.notna(), min_vol_scale)
    vol_scale = np.maximum(vol_scale, min_vol_scale)

    # 以 high/low 相对 close 的偏离，代理主动买入/卖出侧的后悔偏离距离
    buy_dist = (high - close).clip(lower=0)
    sell_dist = (close - low).clip(lower=0)

    z_buy = safe_div(buy_dist, vol_scale)
    z_sell = safe_div(sell_dist, vol_scale)

    # 软阈值截断：phi(z) = k_cap * tanh(z / k_cap)
    phi_buy = k_cap * np.tanh(z_buy / k_cap)
    phi_sell = k_cap * np.tanh(z_sell / k_cap)

    # 前景理论非对称效用映射
    buy_utility = lambda_buy * np.power(phi_buy, beta_buy)
    sell_utility = lambda_sell * np.power(phi_sell, beta_sell)

    # 量能压缩权重：w = min(1, (amount / Q_ref)^gamma)
    q_ref = roller_quantile(value, q_ref_quantile, q_ref_lookback, 1, 'rolling')
    amt_ratio = safe_div(value, q_ref)
    weight = np.power(amt_ratio, gamma)
    weight = np.minimum(weight, 1.0)

    # 日内窗口退化到日线代理，使用 weriod 做行为压力滚动聚合
    buy_pressure = weight * buy_utility
    sell_pressure = weight * sell_utility

    denom = roller_sum(weight, weriod, 1, method) + eps
    r_buy = safe_div(roller_sum(buy_pressure, weriod, 1, method), denom)
    r_sell = safe_div(roller_sum(sell_pressure, weriod, 1, method), denom)

    alpha_raw = r_buy - r_sell

    # 框架强制最终平滑
    alpha = roller_mean(alpha_raw, window, 1, method)
    return alpha
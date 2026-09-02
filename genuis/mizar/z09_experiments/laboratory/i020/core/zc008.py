import pdb
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def zc008(close, high, low, volume, window, fast, slow, ewm=False):
    method = 'ewm' if ewm else 'rolling'

    # ------------------------------------------------------------
    # Daily raw convexity proxy from OHLCV.
    # loc_sell / loc_buy are intraday location proxies of sell/buy
    # side pressure. The original order-book convexity is not
    # directly representable in the wide OHLCV container.
    # ------------------------------------------------------------
    price_range = high - low
    loc_buy = safe_div(close - low, price_range)
    loc_sell = safe_div(high - close, price_range)
    conv_raw = (loc_sell - loc_buy) * volume

    # ------------------------------------------------------------
    # Daily realized vol, used only through lagged/causal windows.
    # ------------------------------------------------------------
    ret = safe_shift(close, 1)
    rv = ret.abs()

    # ------------------------------------------------------------
    # Causal volatility regime.
    # q40 / q60 are rolling quantiles over RV[t-fast ... t-1].
    # ------------------------------------------------------------
    rv_prev = rv.shift(1)
    q40 = roller_quantile(rv_prev, 0.4, fast, 1, 'rolling')
    q60 = roller_quantile(rv_prev, 0.6, fast, 1, 'rolling')

    valid_q = q40.notna() & q60.notna()
    # state code: 0 = L, 1 = M, 2 = H
    raw_state = pd.DataFrame(1.0, index=rv.index, columns=rv.columns)
    raw_state = raw_state.mask(rv_prev > q60, 2.0)
    raw_state = raw_state.mask(rv_prev <= q40, 0.0)
    raw_state = raw_state.where(valid_q)

    # ------------------------------------------------------------
    # 3-day majority smoothing over raw_state_{t-3}, raw_state_{t-2},
    # raw_state_{t-1}; fallback is raw_state_{t-1}.
    # ------------------------------------------------------------
    r1 = raw_state.shift(1)
    r2 = raw_state.shift(2)
    r3 = raw_state.shift(3)

    cnt_l = (((r1 == 0).astype(float)) + ((r2 == 0).astype(float)) + ((r3 == 0).astype(float)))
    cnt_m = (((r1 == 1).astype(float)) + ((r2 == 1).astype(float)) + ((r3 == 1).astype(float)))
    cnt_h = (((r1 == 2).astype(float)) + ((r2 == 2).astype(float)) + ((r3 == 2).astype(float)))

    maj_l = (cnt_l >= 2.0) & (cnt_l > cnt_m) & (cnt_l > cnt_h)
    maj_m = (cnt_m >= 2.0) & (cnt_m > cnt_l) & (cnt_m > cnt_h)
    maj_h = (cnt_h >= 2.0) & (cnt_h > cnt_l) & (cnt_h > cnt_m)

    state = pd.DataFrame(1.0, index=rv.index, columns=rv.columns)
    state = state.mask(maj_l, 0.0)
    state = state.mask(maj_h, 2.0)
    state = state.where(maj_l | maj_m | maj_h, r1)

    # ------------------------------------------------------------
    # State-conditional moments for raw convexity proxy.
    # Use only u < t, i.e. shifted F and shifted state.
    # ------------------------------------------------------------
    F = conv_raw
    F_prev = F.shift(1)
    state_prev = state.shift(1)
    valid_F = F_prev.notna() & state_prev.notna()

    F_l = F_prev.where((state_prev == 0) & valid_F)
    F_m = F_prev.where((state_prev == 1) & valid_F)
    F_h = F_prev.where((state_prev == 2) & valid_F)

    obs_l = roller_sum(((state_prev == 0) & valid_F).astype(float), slow, 1, method)
    obs_m = roller_sum(((state_prev == 1) & valid_F).astype(float), slow, 1, method)
    obs_h = roller_sum(((state_prev == 2) & valid_F).astype(float), slow, 1, method)

    mu_all = roller_mean(F_prev.where(valid_F), slow, 1, method)
    sigma_all = roller_std(F_prev.where(valid_F), slow, 1, method)

    mu_l = roller_mean(F_l, slow, 1, method)
    mu_m = roller_mean(F_m, slow, 1, method)
    mu_h = roller_mean(F_h, slow, 1, method)

    sigma_l = roller_std(F_l, slow, 1, method)
    sigma_m = roller_std(F_m, slow, 1, method)
    sigma_h = roller_std(F_h, slow, 1, method)

    MIN_OBS = 20
    EPS = 1e-8

    mu_l = mu_l.where(obs_l >= MIN_OBS, mu_all)
    mu_m = mu_m.where(obs_m >= MIN_OBS, mu_all)
    mu_h = mu_h.where(obs_h >= MIN_OBS, mu_all)

    sigma_l = sigma_l.where(obs_l >= MIN_OBS, sigma_all)
    sigma_m = sigma_m.where(obs_m >= MIN_OBS, sigma_all)
    sigma_h = sigma_h.where(obs_h >= MIN_OBS, sigma_all)

    sigma_l = sigma_l.where(sigma_l > EPS, EPS)
    sigma_m = sigma_m.where(sigma_m > EPS, EPS)
    sigma_h = sigma_h.where(sigma_h > EPS, EPS)

    z_l = safe_div(F - mu_l, sigma_l)
    z_m = safe_div(F - mu_m, sigma_m)
    z_h = safe_div(F - mu_h, sigma_h)

    # ------------------------------------------------------------
    # Soft fusion with lagged log-RV.
    # ------------------------------------------------------------
    log_rv = self_log(rv.where(rv > EPS, EPS))
    v_prev = log_rv.shift(1)
    valid_v = v_prev.notna() & state_prev.notna()

    v_l = v_prev.where((state_prev == 0) & valid_v)
    v_m = v_prev.where((state_prev == 1) & valid_v)
    v_h = v_prev.where((state_prev == 2) & valid_v)

    cnt_v_l = roller_sum(((state_prev == 0) & valid_v).astype(float), fast, 1, 'rolling')
    cnt_v_m = roller_sum(((state_prev == 1) & valid_v).astype(float), fast, 1, 'rolling')
    cnt_v_h = roller_sum(((state_prev == 2) & valid_v).astype(float), fast, 1, 'rolling')

    c_global = roller_mean(v_prev, fast, 1, 'rolling')
    c_l = roller_mean(v_l, fast, 1, 'rolling')
    c_m = roller_mean(v_m, fast, 1, 'rolling')
    c_h = roller_mean(v_h, fast, 1, 'rolling')

    c_l = c_l.where(cnt_v_l > 0, c_global)
    c_m = c_m.where(cnt_v_m > 0, c_global)
    c_h = c_h.where(cnt_v_h > 0, c_global)

    SOFT_LAG_SMOOTH_DAYS = 3
    V = roller_mean(v_prev, SOFT_LAG_SMOOTH_DAYS, 1, method)

    BAND_SCALE = 0.5
    h_raw = BAND_SCALE * (c_h - c_l)
    h = h_raw.where(h_raw > EPS, EPS)

    d_l = V - c_l
    d_m = V - c_m
    d_h = V - c_h
    h2 = h * h

    w_l = np.exp(-0.5 * safe_div(d_l * d_l, h2))
    w_m = np.exp(-0.5 * safe_div(d_m * d_m, h2))
    w_h = np.exp(-0.5 * safe_div(d_h * d_h, h2))

    weighted = w_l * z_l + w_m * z_m + w_h * z_h
    weight_total = w_l + w_m + w_h
    weight_total = weight_total.where(weight_total > EPS, EPS)

    Fz = safe_div(weighted, weight_total)

    # ------------------------------------------------------------
    # Final smoothing mandated by the framework.
    # ------------------------------------------------------------
    alpha = roller_mean(Fz, window, 1, method)
    return alpha
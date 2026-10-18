"""
zc003.py — VWPIN自适应Zeta收益冲击因子 (O1)

核心逻辑：
1. 分钟对数收益率 r_t = safe_shift(close)
2. 因果条件波动率 sigma_t（仅使用截至 t-1 的平方收益）
3. 标准化收益 z_t = r_t / sigma_t
4. 滚动成交量加权Hill尾部指数 alpha_t（滞后1bar，严格因果）
5. alpha_t -> zeta_t 映射 + 指数平滑 + 单bar变化限幅
6. 成交量加权 |z_i|^{zeta_t} 的 factor_window 滚动均值
7. 框架强制最终平滑
"""

import pdb
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *


def zc003(close, volume, window, fast, slow, weriod, ewm=False,
          warmup_bars=100, min_vol=1e-8,
          tail_threshold_quantile=0.90, min_tail_count=30,
          default_tail_alpha=4.0, alpha_min=1.0, alpha_max=10.0,
          alpha_to_zeta_offset=10.0, zeta_min=2.0, zeta_max=8.0,
          zeta_smooth_eta=0.1, zeta_change_limit=0.1):
    """
    Parameters
    ----------
    close : DataFrame
        分钟收盘价宽表。
    volume : DataFrame
        分钟成交量宽表。
    window : int
        框架要求：return 前最终平滑窗口。
    fast : int
        factor_window，最终因子F_t的滚动窗口长度。
    slow : int
        tail_window，尾部指数估计的滚动窗口长度。
    weriod : int
        条件波动率估计窗口（EWMA span 或 rolling window）。
    ewm : bool
        True 使用 ewm，False 使用 rolling。
    """
    method = 'ewm' if ewm else 'rolling'

    # Step 1: 分钟对数收益率
    rets = safe_shift(close, 1)
    rets2 = rets ** 2

    # Step 2: 因果条件波动率
    # 对滞后1bar的平方收益做滚动/EWMA均值，sigma_t 不使用当前收益信息
    vol_minp = min(int(weriod), int(warmup_bars))
    vol2 = roller_mean(rets2.shift(1), weriod, vol_minp, method)
    vol2 = vol2.clip(lower=min_vol ** 2)
    vol = vol2 ** 0.5

    # 标准化收益
    z = safe_div(rets, vol)
    abs_z = z.abs()

    # Step 3: 滚动成交量加权Hill尾部指数估计
    u = roller_quantile(abs_z, tail_threshold_quantile, slow, 1, 'rolling')
    in_tail = (abs_z > u).astype(float)

    tail_vol = volume.where(in_tail > 0, 0.0)
    tail_log = safe_log(abs_z, u).where(in_tail > 0)

    num_tail = roller_sum(tail_vol * tail_log, slow, 1, 'rolling')
    den_tail = roller_sum(tail_vol, slow, 1, 'rolling')
    tail_cnt = roller_sum(in_tail, slow, 1, 'rolling')

    alpha_inv = safe_div(num_tail, den_tail)
    alpha_raw = 1.0 / alpha_inv

    # 尾部样本不足时回退默认alpha；整体滞后1bar确保因果性
    alpha = alpha_raw.where(tail_cnt >= min_tail_count, default_tail_alpha)
    alpha = alpha.clip(lower=alpha_min, upper=alpha_max).shift(1)

    # Step 4: 自适应 zeta 映射 + 平滑 + 单bar变化限幅
    zeta_raw = (alpha_to_zeta_offset - alpha).clip(
        lower=zeta_min, upper=zeta_max)
    zeta_span = max(int(2.0 / zeta_smooth_eta - 1.0), 3)
    zeta_smooth_raw = roller_mean(zeta_raw, zeta_span, 1, method)

    # 向量化近似实现单bar变化限制（框架禁止循环与cumsum）
    zeta_prev = zeta_smooth_raw.shift(1)
    zeta_delta = (zeta_smooth_raw - zeta_prev).clip(
        lower=-zeta_change_limit, upper=zeta_change_limit)
    zeta_smooth = zeta_prev + zeta_delta

    # Step 5: 成交量加权极端冲击因子（factor_window 滚动窗口）
    pow_z = abs_z ** zeta_smooth
    f_num = roller_sum(volume * pow_z, fast, 1, 'rolling')
    f_den = roller_sum(volume, fast, 1, 'rolling')
    alpha_raw_f = safe_div(f_num, f_den)

    # Step 6: 框架强制最终平滑
    alpha = roller_mean(alpha_raw_f, window, 1, method)
    return alpha
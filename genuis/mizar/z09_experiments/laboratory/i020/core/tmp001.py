"""
tmp001.py - Core engine for OI state strength factor.
"""
import pdb
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def tmp001(close, openint, window, fast, slow, ewm=False,
           gamma=1.0, alpha_coef=0.5, lam=0.05, persistence_max=50):
    method = 'ewm' if ewm else 'rolling'
    # Step 1: 收益与持仓量变化
    rets = safe_shift(close, 1)
    d_oi = openint - openint.shift(1)

    # Step 2: 滚动标准化 z-score
    r_mean = roller_mean(rets, fast, 1, method)
    r_sd = roller_std(rets, fast, 1, method)
    r_std = safe_div(rets - r_mean, r_sd)

    oi_mean = roller_mean(d_oi, fast, 1, method)
    oi_sd = roller_std(d_oi, fast, 1, method)
    oi_std = safe_div(d_oi - oi_mean, oi_sd)

    # Step 3: 四象限状态分类
    sign_r = (r_std > 0).astype(float) - (r_std < 0).astype(float)
    sign_oi = (oi_std > 0).astype(float) - (oi_std < 0).astype(float)
    sign_state = (sign_r * sign_oi).where(r_std.notna() & oi_std.notna())

    state = (sign_r > 0).astype(int) * 2 + (sign_oi > 0).astype(int)
    state = state.where(r_std.notna() & oi_std.notna())

    # Step 4: 状态持续性 persistence
    pos = pd.DataFrame(np.arange(len(close))[:, None],
                       index=close.index, columns=close.columns)

    state_filled = state.fillna(-999)
    change = (state_filled != state_filled.shift(1)).astype(float)
    change_pos = pos.where(change > 0)

    last_change = roller_max(change_pos, persistence_max, 1, 'rolling')
    last_change = last_change.where(last_change.notna(), 0)

    run_len = pos - last_change + 1
    run_len = run_len.where(run_len <= persistence_max, persistence_max)
    p_norm = run_len / persistence_max

    # Step 5: 强度权重，tanh 非线性压缩
    w = np.tanh(gamma * oi_std.abs())

    # Step 6: 状态强度编码
    f_state = sign_state * w * (1.0 + alpha_coef * p_norm)

    # Step 7: 指数衰减加权聚合，over past slow bars
    weighted = f_state.copy()
    for i in range(1, slow):
        weighted = weighted + f_state.shift(i) * np.exp(-lam * i)

    # 强制最终平滑
    alpha = roller_mean(weighted, window, 1, method)
    return alpha
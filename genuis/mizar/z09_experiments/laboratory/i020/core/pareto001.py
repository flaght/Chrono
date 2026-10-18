import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def pareto001(volume, window, fast, slow, ewm=False, q_low=0.1, q_high=0.9, epsilon=1e-6):
    method = 'ewm' if ewm else 'rolling'
    # 步骤1：计算分位数（仅支持 rolling，固定使用）
    Q_low  = roller_quantile(volume, q_low,  fast, 1, 'rolling')
    Q_high = roller_quantile(volume, q_high, fast, 1, 'rolling')
    # 处理低分位数为0的情况，替换为极小正数
    Q_low_safe = Q_low.where(Q_low > 0, epsilon)
    # 步骤2：分位数比率
    R = safe_div(Q_high, Q_low_safe)
    # 步骤3：滚动标准化（固定 rolling）
    mu_R    = roller_mean(R, slow, 1, 'rolling')
    sigma_R = roller_std(R, slow, 1, 'rolling')
    # z-score，注意 sigma=0 时置为 0
    alpha_raw = safe_div(R - mu_R, sigma_R)
    alpha_raw = alpha_raw.where(sigma_R != 0, 0)
    # 最终平滑
    alpha = roller_mean(alpha_raw, window, 1, method)
    return alpha
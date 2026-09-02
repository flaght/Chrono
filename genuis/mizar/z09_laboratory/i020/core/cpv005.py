"""cpv005 core module"""
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *


def cpv005(close, volume, openint, window=5, weriod=20, h=0.5, ewm=False):
    """
    动态自适应的价格-持仓量相关系数因子（简化版）
    使用标准化残差作为 CUSUM 变点检测门控，动态切换短/长窗口相关系数。
    """
    method = 'ewm' if ewm else 'rolling'

    # 1. 价格对数收益率
    ret = safe_shift(close, 1)

    # 2. 持仓量差分
    delta_oi = openint - openint.shift(1)

    # 3. 修正持仓量（累计和）
    oi_adj = roller_sum(delta_oi, weriod, 1, method)  #delta_oi.cumsum()

    # 4. 滚动均值与标准差
    mean_oi = roller_mean(oi_adj, weriod, 1, method)
    std_oi = roller_std(oi_adj, weriod, 1, method)

    # 5. 标准化残差（安全除法）
    z = safe_div(oi_adj - mean_oi, std_oi)

    # 6. 变点标志（|z| > h 视为变点）
    flag = (abs(z) > h).astype(float)

    # 7. 短窗口（2）与长窗口（weriod）相关系数
    corr_short = roller_corr(ret, delta_oi, 2, 1, method)
    corr_long = roller_corr(ret, delta_oi, weriod, 1, method)

    # 8. 根据变点标志动态选择
    alpha = corr_short.where(flag > 0.5, corr_long)

    # 9. 最终平滑
    alpha = roller_mean(alpha, window, 1, method)

    return alpha

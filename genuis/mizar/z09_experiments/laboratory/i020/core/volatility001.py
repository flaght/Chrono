"""
volatility001_core.py — 基于高阶矩与波动率残差的因子
"""
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *


def volatility001(close, window, weriod, ewm=False):
    """
    因子计算：稳健偏度/峰度与市场波动率（VIX代理、RV）的残差组合
    参数:
        close   : DataFrame, 宽表，收盘价
        window  : int, 最终平滑窗口
        weriod  : int, 滚动回归窗口（观测数）
        ewm     : bool, 是否使用指数加权（仅用于最终平滑）
    """
    method = 'ewm' if ewm else 'rolling'

    # 1. 分钟对数收益率
    rets = safe_shift(close, 1)

    # 2. 高阶矩计算窗口（固定60，说明书要求）
    mom_w = 60

    # 3. 计算分位数
    Q1 = roller_quantile(rets, 0.25, mom_w, 1, 'rolling')
    Q2 = roller_quantile(rets, 0.5,  mom_w, 1, 'rolling')
    Q3 = roller_quantile(rets, 0.75, mom_w, 1, 'rolling')
    P10 = roller_quantile(rets, 0.1,  mom_w, 1, 'rolling')
    P90 = roller_quantile(rets, 0.9,  mom_w, 1, 'rolling')

    # 4. 稳健偏度与峰度（分位数公式）
    skew_num = (Q3 - Q2) - (Q2 - Q1)
    skew_den = Q3 - Q1
    skew = safe_div(skew_num, skew_den)

    kurt_num = Q3 - Q1
    kurt_den = 2 * (P90 - P10)
    kurt = safe_div(kurt_num, kurt_den)

    # 5. 波动率指标
    VIX = roller_std(rets, mom_w, 1, 'rolling')          # 代理VIX
    RV = (roller_sum(rets**2, 5, 1, 'rolling'))**0.5     # 5分钟已实现波动率

    # 6. 滚动回归窗口
    reg_w = weriod

    # 7. 对 skew 用 VIX 做单变量回归，取残差
    mean_skew = roller_mean(skew, reg_w, 1, method)
    mean_vix  = roller_mean(VIX,  reg_w, 1, method)
    cov_skew_vix = roller_cov(skew, VIX, reg_w, 1, 'rolling')
    var_vix      = roller_var(VIX,  reg_w, 1, 'rolling')
    slope_skew   = safe_div(cov_skew_vix, var_vix)
    res_skew = skew - (mean_skew + slope_skew * (VIX - mean_vix))

    # 8. 对 kurt 用 RV 做单变量回归，取残差
    mean_kurt = roller_mean(kurt, reg_w, 1, method)
    mean_rv   = roller_mean(RV,   reg_w, 1, method)
    cov_kurt_rv = roller_cov(kurt, RV, reg_w, 1, 'rolling')
    var_rv      = roller_var(RV,   reg_w, 1, 'rolling')
    slope_kurt  = safe_div(cov_kurt_rv, var_rv)
    res_kurt = kurt - (mean_kurt + slope_kurt * (RV - mean_rv))

    # 9. 组合残差
    res_comb = res_skew + res_kurt

    # 10. 最终平滑（框架强制）
    alpha = roller_mean(res_comb, window, 1, method)
    return alpha
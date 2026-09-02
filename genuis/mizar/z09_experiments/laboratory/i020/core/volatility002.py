# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *


def volatility002(close, window, weriod, lookback=5, lambda_param=1.0,
                  zscore_threshold=3, mad_scale=1.4826, ewm=False):
    """
    基于日内形状预期偏离的因子（HF-SED-Panel 简化版）
    
    参数
    ----
    close : pd.DataFrame
        宽表，索引为时间，列为标的代码
    window : int
        最终平滑窗口
    weriod : int
        计算分位数/偏度/峰度的滚动窗口
    lookback : int
        计算预期形状与MAD的滚动窗口
    lambda_param : float
        峰度分量的合并权重
    zscore_threshold : float
        残差标准化后的截断阈值（绝对值超过则置0）
    mad_scale : float
        MAD转标准差的常数
    ewm : bool
        是否使用指数加权（仅对支持ewm的算子生效，quantile固定rolling）
    
    返回
    ----
    alpha : pd.DataFrame
        连续浮点数的因子值，已做最终平滑
    """
    method = 'ewm' if ewm else 'rolling'
    
    # 收益率（对数）
    rets = safe_shift(close, 1)
    
    # 稳健偏度需要的分位数
    P10 = roller_quantile(rets, 0.10, weriod, 1, 'rolling')
    P50 = roller_quantile(rets, 0.50, weriod, 1, 'rolling')
    P90 = roller_quantile(rets, 0.90, weriod, 1, 'rolling')
    
    # 稳健峰度需要的分位数
    P25 = roller_quantile(rets, 0.25, weriod, 1, 'rolling')
    P75 = roller_quantile(rets, 0.75, weriod, 1, 'rolling')
    P2_5 = roller_quantile(rets, 0.025, weriod, 1, 'rolling')
    P97_5 = roller_quantile(rets, 0.975, weriod, 1, 'rolling')
    
    # 稳健偏度： (P90 - P50 - (P50 - P10)) / (P90 - P10)
    skew_robust = safe_div(P90 - P50 - (P50 - P10), P90 - P10)
    
    # 稳健峰度： (P97.5 - P2.5) / (P75 - P25) - 2.91
    kurt_robust = safe_div(P97_5 - P2_5, P75 - P25) - 2.91
    
    # 预期形状：滚动均值
    expected_skew = roller_mean(skew_robust, lookback, 1, method)
    expected_kurt = roller_mean(kurt_robust, lookback, 1, method)
    
    # 残差
    resid_skew = skew_robust - expected_skew
    resid_kurt = kurt_robust - expected_kurt
    
    # 残差的中位数
    med_skew = roller_median(resid_skew, lookback, 1, 'rolling')
    med_kurt = roller_median(resid_kurt, lookback, 1, 'rolling')
    
    # MAD（中位数绝对偏差）
    mad_skew = roller_median((resid_skew - med_skew).abs(), lookback, 1, 'rolling')
    mad_kurt = roller_median((resid_kurt - med_kurt).abs(), lookback, 1, 'rolling')
    
    # 标准化（MAD转标准差）
    z_skew = safe_div(resid_skew - med_skew, mad_scale * mad_skew)
    z_kurt = safe_div(resid_kurt - med_kurt, mad_scale * mad_kurt)
    
    # 剔除异常残差（|z| > zscore_threshold 置0）
    z_skew = z_skew.where(z_skew.abs() <= zscore_threshold, 0)
    z_kurt = z_kurt.where(z_kurt.abs() <= zscore_threshold, 0)
    
    # 合并偏度与峰度分量
    alpha = z_skew + lambda_param * z_kurt
    
    # 最终平滑（框架硬性要求）
    alpha = roller_mean(alpha, window, 1, method)
    return alpha
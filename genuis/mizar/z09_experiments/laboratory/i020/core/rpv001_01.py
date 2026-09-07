"""
rpv001_01_core.py — 基于 RPV（相对价格波动）的背离调整动量因子
"""
import numpy as np
import pandas as pd
from lumina.impulse.fixed import *

def rpv001_01(close, high, low, window, weriod, ewm=False):
    """
    计算因子：
    1. RPV = (high - low) / close
    2. EMA平滑RPV（span=3，近似半衰期3）
    3. 滚动z-score
    4. 背离调整（价格动量与RPV动量背离时衰减）
    5. 最终平滑
    
    Parameters
    ----------
    close : pd.DataFrame
    high : pd.DataFrame
    low : pd.DataFrame
    window : int, 最终平滑周期
    weriod : int, 滚动统计周期
    ewm : bool, 是否使用指数加权滚动
    
    Returns
    -------
    pd.DataFrame, 连续alpha值
    """
    method = 'ewm' if ewm else 'rolling'
    
    # 1. RPV
    rpv = safe_div(high - low, close)
    
    # 2. EMA平滑（span=3）
    rpv_smooth = roller_mean(rpv, 3, 1, 'ewm')
    
    # 3. 滚动z-score
    rpv_mean = roller_mean(rpv_smooth, weriod, 1, method)
    rpv_std = roller_std(rpv_smooth, weriod, 1, method)
    z_score = safe_div(rpv_smooth - rpv_mean, rpv_std)
    
    # 4. 背离调整
    price_mom = close - close.shift(weriod)          # 价格动量
    rpv_mom = rpv_smooth - rpv_smooth.shift(weriod)  # RPV动量
    divergence = price_mom * rpv_mom                 # 同号为正，异号为负（背离）
    # 连续调整因子：背离时(divergence<0) tanh为负，adjust<0.5，缩小|alpha|
    adjust = 0.5 + 0.5 * np.tanh(divergence)
    alpha = z_score * adjust
    
    # 5. 最终平滑
    alpha = roller_mean(alpha, window, 1, method)
    return alpha
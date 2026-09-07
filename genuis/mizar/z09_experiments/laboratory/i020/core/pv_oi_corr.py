import pdb
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *


def pv_oi_corr(close, openint, volume, window, weriod, ewm=False):
    """
    基于修正持仓量的价量相关性因子（PV）
    Parameters
    ----------
    close : pd.DataFrame
        收盘价
    openint : pd.DataFrame
        持仓量
    volume : pd.DataFrame
        成交量
    window : int
        最终平滑窗口
    weriod : int
        滚动统计及成交量加权窗口
    ewm : bool
        是否使用 ewm 平滑，否则 rolling
    Returns
    -------
    alpha : pd.DataFrame
        连续浮点数因子值
    """
    method = 'ewm' if ewm else 'rolling'

    # 成交量加权平均
    vol_mean = roller_mean(volume, weriod, 1, method)

    # 修正持仓量：成交量加权分配，放大交易活跃日的持仓量权重
    adj_oi = openint * safe_div(volume, vol_mean)

    # 价格与修正持仓量的滚动相关系数
    pv = roller_corr(close, adj_oi, weriod, 1, method)

    # 最终平滑输出
    alpha = roller_mean(pv, window, 1, method)
    return alpha
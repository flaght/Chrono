# -*- encoding:utf-8 -*-
import numpy as np
from lumina.impulse.fixed import *


def ki004(open, close, volume, window, weriod, ewm=False):
    """
    RPV因子 - 价量相关性反转因子 (Renewed Price-Volume Correlation)

    来源: 东吴证券《选股因子系列研究(七十八)》

    原理:
        1. CCOIV: 日内收益与日内成交量的相关系数
           - 衡量日内价量关系，反映反转效应 (负IC)
        2. COV: 隔夜收益与前期成交量的相关系数
           - 衡量隔夜价量关系，反映动量效应 (正IC)
        3. RPV = CCOIV_norm - COV_norm
           - 两者截面标准化后相减，组合反转与动量信号

    公式:
        intraday_ret = close - open  (日内收益)
        overnight_ret = open - close.shift(1)  (隔夜收益)
        CCOIV = Corr(intraday_ret, volume, weriod)
        COV = Corr(overnight_ret, volume.shift(1), weriod)
        RPV = zscore(CCOIV) - zscore(COV)  (截面标准化)

    参数:
        open: 开盘价 DataFrame
        close: 收盘价 DataFrame
        volume: 成交量 DataFrame
        window: 外层平滑窗口
        weriod: 核心计算窗口 (滚动相关系数窗口)
        ewm: 是否使用指数加权

    返回:
        RPV因子值

    信号解读:
        RPV > 0: 反转信号强于动量信号，看涨
        RPV < 0: 动量信号强于反转信号，看跌
    """
    method = 'ewm' if ewm else 'rolling'

    # 1. 计算日内收益 (close - open)
    intraday_ret = close - open

    # 2. 计算隔夜收益 (open - 前一期close)
    overnight_ret = open - close.shift(1)

    # 3. 前一期成交量
    prev_volume = volume.shift(1)

    # 4. 计算 CCOIV: 日内收益与日内成交量的滚动相关系数
    ccoiv = roller_corr(intraday_ret, volume, weriod, weriod, method)

    # 5. 计算 COV: 隔夜收益与前期成交量的滚动相关系数
    cov = roller_corr(overnight_ret, prev_volume, weriod, weriod, method)

    # 6. 截面标准化 (cross-sectional z-score)
    # 对每个时间点，计算所有品种的均值和标准差进行标准化

    ccoiv_mean = roller_mean(ccoiv, weriod, weriod, method)
    ccoiv_std = roller_std(ccoiv, weriod, weriod, method)
    ccoiv_norm = (ccoiv - ccoiv_mean) / (ccoiv_std + 1e-10)
    #ccoiv_mean = ccoiv.mean(axis=1)
    #ccoiv_std = ccoiv.std(axis=1)
    #ccoiv_norm = ccoiv.sub(ccoiv_mean, axis=0).div(ccoiv_std + 1e-10, axis=0)

    cov_mean = roller_mean(cov, weriod, weriod, method)
    cov_std = roller_std(cov, weriod, weriod, method)
    cov_norm = (cov - cov_mean) / (cov_std + 1e-10)
    #cov_mean = cov.mean(axis=1)
    #cov_std = cov.std(axis=1)
    #cov_norm = cov.sub(cov_mean, axis=0).div(cov_std + 1e-10, axis=0)

    # 7. RPV = CCOIV_norm - COV_norm
    # CCOIV 负IC (反转), COV 正IC (动量)
    # 相减后: 高CCOIV低COV -> 高RPV -> 预期上涨
    core1 = ccoiv_norm - cov_norm

    # 8. 最终用 window 做平滑
    alpha = roller_mean(core1, window, window, method)

    return alpha

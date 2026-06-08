"""
kx031 - 寻找业绩与估值的错配，非理性估值溢价因子

研报来源: 寻找业绩与估值的错配，非理性估值溢价因子.pdf
实现状态: generated
数据字段: close, volume
实现说明: 基于价格动量和均值回归信号，寻找业绩与估值的错配机会
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx031(close, volume, weriod, window, ewm):
    """
    寻找业绩与估值的错配，非理性估值溢价因子 (kx031)

    基于价格动量和均值回归信号，寻找业绩表现与估值水平的错配机会。
    当业绩优秀但估值偏低，或业绩一般但估值严重偏低时，获得正向信号。

    参数:
        close: 收盘价 DataFrame
        volume: 成交量 DataFrame
        weriod: 业绩评估周期 (默认20)
        window: 最终平滑窗口 (默认20)
        ewm: 是否使用指数加权

    返回:
        估值错配因子值 (正值表示错配机会)
    """

    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 业绩表现信号 (动量因子近似)
    # 近期价格动量作为业绩表现的代理指标
    momentum_short = roller_mean(close.pct_change(), weriod, weriod, method)
    momentum_long = roller_mean(close.pct_change(), weriod, weriod, method)
    performance_signal = momentum_short - momentum_long  # 动量背离

    # 估值水平信号 (均值回归因子近似)
    # 价格偏离度作为估值水平的代理指标
    price_ma = roller_mean(close, weriod, weriod, method)
    valuation_signal = (close - price_ma) / (price_ma + 1e-8)

    # 成交量确认信号
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_signal = volume / (volume_ma + 1e-8)

    # 业绩与估值错配因子 (单品种时序逻辑)
    # 核心逻辑：基于个股历史数据判断错配机会
    # 业绩优秀：动量信号高于个股历史均值+1倍标准差
    # 估值偏低：价格偏离度低于个股历史均值-1倍标准差
    # 成交量确认：成交量高于个股历史均值

    # 计算个股历史统计 (单品种时序)
    perf_mean = roller_mean(performance_signal, weriod, weriod, method)
    perf_std = roller_std(performance_signal, weriod, weriod, method)
    val_mean = roller_mean(valuation_signal, weriod, weriod, method)
    val_std = roller_std(valuation_signal, weriod, weriod, method)
    vol_mean = roller_mean(volume_signal, weriod, weriod, method)

    # 基于个股历史判断错配 (连续值逻辑)
    # 将离散条件改为连续值，提供更丰富的信息
    
    # 业绩强度：标准化后的业绩信号 (正值表示业绩优秀)
    performance_zscore = (performance_signal - perf_mean) / (perf_std + 1e-8)
    performance_strength = performance_zscore.clip(-3, 3)  # 限制在-3到3之间

    # 估值便宜度：标准化后的估值信号 (负值表示估值偏低，更便宜)
    valuation_zscore = (valuation_signal - val_mean) / (val_std + 1e-8)
    valuation_cheap = -valuation_zscore.clip(-3, 3)  # 取负号，因为便宜是好的

    # 成交量确认：标准化后的成交量信号 (正值表示成交活跃)
    volume_zscore = (volume_signal - vol_mean) / (roller_std(volume_signal, weriod, weriod, method) + 1e-8)
    volume_confirm = volume_zscore.clip(-3, 3)

    # 错配得分 = 业绩强度 × 估值便宜度 × 成交量确认 (连续值相乘)
    mismatch_score = performance_strength * valuation_cheap * volume_confirm

    factor_values = mismatch_score.astype(float)
    # 最终平滑 (window参数仅用于最终平滑, min_periods=1)
    factor_values = roller_mean(factor_values, window, window, method)

    return factor_values

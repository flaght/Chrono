"""
kx020 - 行业轮动因子 (近似实现)

研报来源: 指数增强如何受益于行业轮动.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于股票表现的聚类分析近似行业轮动，通过股票间的相关性和表现差异识别轮动机会
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx020(close, volume, fast, slow, weriod, window, ewm):
    """
    行业轮动因子 (kx020) - 近似实现

    基于指数增强如何受益于行业轮动的策略，通过股票表现聚类分析识别轮动机会。

    核心逻辑:
        1. 计算股票表现的相关性 (近似行业相关性)
        2. 识别表现分化的股票群 (近似行业轮动)
        3. 构建轮动择时信号
        4. 生成行业轮动因子

    因子原理:
        行业轮动 = 表现分化度 × 轮动强度 × 持续性确认
        基于股票表现差异的轮动策略

    参数说明:
    close: 资产收盘价 DataFrame
    volume: 资产成交量 DataFrame
    weriod: 轮动计算周期 (默认20)
    window: 确认窗口 (默认20)
    ewm: 是否使用指数加权

    返回值:
        行业轮动因子值
    """
    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 确保数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算价格收益率
    returns = close.pct_change()

    # 计算个股表现动量 (近似行业轮动中的个股表现)
    returns_ma_short = roller_mean(returns, fast, fast, method)
    returns_ma_long = roller_mean(returns, slow, slow, method)
    momentum_divergence = returns_ma_short - returns_ma_long  # 动量分化信号

    # 计算波动性变化 (轮动时的波动放大)
    volatility = roller_std(returns, weriod, weriod, method)
    volatility_ma = roller_mean(volatility, weriod, weriod, method)
    volatility_change = volatility - volatility_ma

    # 计算成交量相对强度 (轮动时的成交量配合)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_ratio = volume / (volume_ma + 1e-8)
    volume_strength = (volume_ratio - 1).clip(lower=0)  # 成交量放大强度

    # 行业轮动择时信号 (基于个股自身表现)
    rotation_signal = momentum_divergence * volatility_change * volume_strength

    # 轮动持续性确认 (使用个股层面的滚动平均)
    rotation_persistence = roller_mean(rotation_signal.abs(), weriod, weriod, method)

    # 行业轮动因子 (基于个股轮动信号强度)
    factor_values = rotation_signal * rotation_persistence

    # 应用最终平滑
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

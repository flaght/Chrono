"""
kx021 - 景气度轮动因子 (近似实现)

研报来源: 指数研究与指数化投资系列：景气度视角下制造板块内部轮动配置策略.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于股票表现模式的景气度轮动，通过收益动量和波动性变化识别制造板块内部轮动机会
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx021(close, volume, fast, slow, weriod, window, ewm):
    """
    景气度轮动因子 (kx021) - 近似实现

    基于景气度视角下制造板块内部轮动配置策略，通过收益模式识别景气度轮动。

    核心逻辑:
        1. 计算收益动量模式 (近似景气度变化)
        2. 识别波动性周期 (近似景气度周期)
        3. 构建轮动择时信号
        4. 生成景气度轮动因子

    因子原理:
        景气度轮动 = 收益动量 × 波动周期 × 持续性权重
        基于收益模式识别的景气度轮动策略

    参数说明:
    close: 资产收盘价 DataFrame
    volume: 资产成交量 DataFrame
    weriod: 景气度计算周期 (默认20)
    window: 轮动确认窗口 (默认20)
    ewm: 是否使用指数加权

    返回值:
        景气度轮动因子值
    """
    # 参数验证
    if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
        raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 确保数据对齐
    close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算价格收益率
    returns = close.pct_change()

    # 收益动量计算 (近似景气度变化)
    momentum_short = roller_mean(returns, fast, fast, method)
    momentum_long = roller_mean(returns, slow, slow, method)
    momentum_divergence = momentum_short - momentum_long  # 动量分化

    # 波动性周期识别 (近似景气度周期)
    volatility = roller_std(returns, weriod, weriod, method)
    volatility_cycle = (volatility - roller_mean(volatility, weriod*2, weriod*2, method)) / \
                      (roller_std(volatility, weriod*2, weriod*2, method) + 1e-8)

    # 成交量确认 (景气度变化时的成交量配合)
    volume_momentum = roller_mean(volume.pct_change(), weriod, weriod, method)

    # 景气度轮动信号 (基于个股自身景气度)
    momentum_strength = momentum_divergence.abs()  # 个股动量强度
    volatility_strength = volatility_cycle.abs()   # 个股波动周期强度
    volume_strength = volume_momentum.abs()        # 个股成交量动量强度

    prosperity_signal = momentum_strength * volatility_strength * volume_strength

    # 轮动持续性确认
    rotation_persistence = roller_mean(prosperity_signal, weriod, weriod, method)

    # 景气度轮动因子
    factor_values = prosperity_signal * rotation_persistence


    # 应用最终平滑
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

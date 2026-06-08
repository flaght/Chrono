"""
kx001 - 分析师预期修正动量因子

研报来源: 因子深度研究系列：分析师预期修正动量效应选股策略.pdf
实现状态: generated
数据字段: close, volume
"""
import pandas as pd
import numpy as np

from lumina.impulse.fixed import *

def kx001(close, volume, weriod, window, ewm):
    """
    分析师预期修正动量因子 (kx001)

    基于分析师预期修正的动量效应，通过价格变化和成交量确认来衡量预期修正强度。

    核心逻辑:
        1. 计算价格动量变化
        2. 衡量预期修正信号
        3. 结合成交量确认预期修正强度
        4. 生成分析师预期修正动量因子

    因子原理:
        分析师预期修正动量 = 预期修正信号 × 成交量确认系数
        反映了分析师预期调整对股价的持续影响

    参数说明:
    close: 资产收盘价 DataFrame
    volume: 资产成交量 DataFrame
    weriod: 动量计算周期 (默认20)
    window: 信号平滑窗口 (默认20)
    ewm: 是否使用指数加权

    返回值:
        分析师预期修正动量因子值
    """
    method = 'ewm' if ewm else 'rolling'
    
    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 确保数据对齐
    #close, volume = close.align(volume, join='inner')

    # 计算价格收益率
    returns = close.pct_change()

    # 计算价格动量 (预期修正的基础信号)
    price_momentum = roller_mean(returns, weriod, weriod, method)

    # 计算动量变化 (预期修正信号)
    momentum_change = price_momentum - price_momentum.shift(weriod)

    # 成交量确认系数
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_ratio = volume / (volume_ma + 1e-8)

    # 分析师预期修正动量因子
    factor_values = momentum_change * volume_ratio

    # 应用最终平滑
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values
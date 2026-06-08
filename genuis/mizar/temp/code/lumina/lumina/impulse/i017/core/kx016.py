"""
kx016 - 特质波动率因子的重构 (近似实现)

研报来源: 因子选股系列之十二：特质波动率因子的重构.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于日频个股波动率和流动性特征近似特质风险，使用个股权波动性×流动性风险×非系统性权重
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx016(close, volume, weriod, window, ewm):
    """
    特质波动率因子的重构 (kx016) - 近似实现

    基于特质波动率因子的重构，通过个股波动率和流动性特征来近似特质风险因子。

    核心逻辑:
        1. 计算个股权波动性指标
        2. 评估流动性风险因子
        3. 确定非系统性风险权重
        4. 生成特质波动率因子

    因子原理:
        特质波动率因子 = 个股权波动性 × 流动性风险 × 非系统性权重
        基于日频数据近似特质风险特征

    参数说明:
    close: 资产收盘价 DataFrame
    volume: 资产成交量 DataFrame
    weriod: 波动率计算周期 (默认20)
    window: 风险评估窗口 (默认20)
    ewm: 是否使用指数加权

    返回值:
        特质波动率因子值
    """
    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 确保数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算个股权波动性指标
    returns = close.pct_change()
    idiosyncratic_volatility = roller_std(returns, weriod, weriod, method)

    # 评估流动性风险因子 (成交量反转的波动性)
    volume_change = volume.pct_change()
    liquidity_risk = roller_std(volume_change, weriod, weriod, method)

    # 确定非系统性风险权重 (基于成交量稳定性)
    volume_stability = 1 / (liquidity_risk + 1e-8)  # 流动性越稳定，非系统性风险权重越大
    volume_weight = roller_mean(volume_stability, weriod, weriod, method)

    # 特质波动率因子 (近似实现)
    factor_values = idiosyncratic_volatility * liquidity_risk * volume_weight


    # 应用最终平滑
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

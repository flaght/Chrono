"""
kx023 - Barra波动率因子 (近似实现)

研报来源: Barra模型专题报告（一）：波动率因子.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于价格波动率的Barra风格因子实现，使用历史波动率作为风险度量
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx023(close, volume, weriod, window, ewm):
    """
    Barra波动率因子 (kx023) - 近似实现

    基于Barra风险模型的波动率因子，通过资产的历史波动率度量系统性风险。

    核心逻辑:
        1. 计算资产收益率的波动率
        2. 标准化波动率度量
        3. 构建波动率因子
        4. 生成风险因子值

    因子原理:
        波动率因子 = 标准化波动率度量
        衡量资产的价格波动性和系统性风险

    参数说明:
    close: 资产收盘价 DataFrame
    volume: 资产成交量 DataFrame
    weriod: 波动率计算周期 (默认20)
    window: 标准化窗口 (默认20)
    ewm: 是否使用指数加权

    返回值:
        Barra波动率因子值
    """
    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 确保数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算收益率
    returns = close.pct_change()

    # 计算波动率 (Barra波动率因子核心)
    volatility = roller_std(returns, weriod, weriod, method)

    # 计算成交量调整的波动率 (考虑流动性影响)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_adjustment = volume / (volume_ma + 1e-8)
    adjusted_volatility = volatility * (1 + volume_adjustment)

    # 波动率因子 (标准化处理)
    factor_values = adjusted_volatility


    # 应用最终平滑
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

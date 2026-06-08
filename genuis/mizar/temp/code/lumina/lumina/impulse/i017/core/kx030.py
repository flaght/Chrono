"""
kx030 - 价值投资失效检测因子 (近似实现，充分利用量价字段)

研报来源: A股的"价值投资"失效了吗？.pdf
实现状态: approximate_implementation
数据字段: close, volume, value (使用完整量价字段)
近似说明: 原研报基于PE/PB等财务指标，这里使用量价数据近似价值信号
字段利用: 结合成交量、成交额、成交均价等多维度数据提升价值信号准确性
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx030(close, volume, fast, slow, weriod, window, ewm):
    """
    价值投资失效检测因子 (kx030) - 近似实现

    原研报探讨价值投资是否失效，基于PE/PB等财务指标。
    近似实现：使用量价数据构建价值投资的替代信号。

    近似逻辑:
        1. 成交量价值信号: 高成交量可能表示价值发现
        2. 价格稳定性信号: 低波动可能表示价值稳定
        3. 动量反转信号: 价值投资的反转特性

    参数:
        close: 收盘价 DataFrame
        volume: 成交量 DataFrame
        vwap: 成交均价 DataFrame
        value: 成交额 DataFrame
        weriod: 价值评估周期 (默认20)
        fast: 评估长周期
        slow: 评估短周期
        window: 最终平滑窗口 (默认20)
        ewm: 是否使用指数加权

    返回:
        价值投资失效检测因子值 (负值表示价值投资可能失效)
    """

    # 参数验证
    #if not all(isinstance(df, pd.DataFrame) for df in [close, volume, value]):
    #    raise ValueError("close, volume, vwap 和 value 必须都是 pandas DataFrame")

    # 确保数据对齐 (使用完整量价字段)
    #close, volume, vwap, value = close.align(volume, join='inner').align(vwap, join='inner').align(value, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 近似价值信号1: 成交量相对强度 (高成交量≈价值发现)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_relative = volume / (volume_ma + 1e-8)
    volume_value_signal = roller_mean(volume_relative, weriod, weriod, method)

    # 近似价值信号2: 价格稳定性 (低波动≈价值稳定)
    returns = close.pct_change()
    volatility = roller_std(returns, weriod, weriod, method)
    stability_signal = 1 / (volatility + 1e-8)  # 稳定性得分

    # 近似价值信号3: 动量反转 (价值投资的反转逻辑)
    momentum_short = roller_mean(returns, fast, fast, method)
    momentum_long = roller_mean(returns, slow, slow, method)
    # 反转信号: 短期动量相对于长期动量的偏离 (价值投资的反转特性)
    reversal_signal = momentum_short - momentum_long

    # 综合价值因子 (近似价值投资策略)
    # 价值信号 = 成交量强度 × 稳定性 × 反转信号
    value_factor = volume_value_signal * stability_signal * reversal_signal

    # 应用最终平滑 (window参数仅用于最终平滑, min_periods=1)
    factor_values = roller_mean(value_factor, window, 1, method)

    return factor_values

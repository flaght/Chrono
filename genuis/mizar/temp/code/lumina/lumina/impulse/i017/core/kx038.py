"""
kx038 - 趋势反转因子 (近似实现)

研报来源: 量化因子掘金系列（一），一个趋势反转因子的构建.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于日频数据构建趋势反转因子，使用价格动量反转和成交量确认
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx038(close, volume, fast, slow, weriod, window, ewm):
    """
    趋势反转因子 (kx038) - 近似实现

    量化因子掘金系列，一个趋势反转因子的构建。
    基于日频数据实现趋势反转信号检测。

    近似逻辑:
        1. 计算趋势强度 (价格动量)
        2. 识别反转信号 (动量衰减)
        3. 评估成交量确认 (反转可靠性)

    参数:
        close: 收盘价 DataFrame
        volume: 成交量 DataFrame
        fast: 短期趋势评估周期 (默认5)
        slow: 长期趋势评估周期 (默认20)
        window: 最终平滑窗口 (默认20)
        ewm: 是否使用指数加权

    返回:
        趋势反转因子值 (正值表示反转机会)
    """

    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算收益率
    returns = close.pct_change()

    # 趋势强度 (短期vs长期动量)
    momentum_short = roller_mean(returns, fast, fast, method)
    momentum_long = roller_mean(returns, slow, slow, method)
    trend_strength = momentum_short - momentum_long

    # 动量衰减信号 (趋势反转迹象)
    momentum_change = trend_strength.diff()
    momentum_decay = (-momentum_change).where(momentum_change < 0, 0)  # 只关注衰减

    # 成交量放大确认 (反转的成交量支持)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_ratio = volume / (volume_ma + 1e-8)

    # 基于个股历史统计的成交量确认 (单品种逻辑)
    volume_ratio_mean = roller_mean(volume_ratio, weriod, weriod, method)
    volume_ratio_std = roller_std(volume_ratio, weriod, weriod, method)
    volume_confirmation = (volume_ratio > volume_ratio_mean + volume_ratio_std).astype(int)

    # 价格波动性确认 (反转期的波动放大)
    volatility = roller_std(returns, weriod, weriod, method)
    vol_ma = roller_mean(volatility, weriod, weriod, method)
    volatility_confirmation = (volatility > vol_ma).astype(int)

    # 趋势反转因子 = 动量衰减 × 成交量确认 × 波动确认
    reversal_factor = momentum_decay * volume_confirmation * volatility_confirmation

    # 标准化处理
    factor_values = reversal_factor

    # 最终平滑 (window参数仅用于最终平滑, min_periods=1)
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

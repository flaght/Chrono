"""
kx043 - 成长周期共振因子 (近似实现)

研报来源: 开源量化评论（107）：成长与周期共振，基于业绩增速与景气定位的双因子协同.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于日频数据近似成长周期共振因子，使用价格动量和成交量变化
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx043(close, volume, weriod, window, ewm):
    """
    成长周期共振因子 (kx043) - 近似实现

    开源量化评论（107）：成长与周期共振，基于业绩增速与景气定位的双因子协同。
    基于日频数据近似成长周期共振逻辑。

    近似逻辑:
        1. 成长动量信号 (中长期上涨趋势)
        2. 周期景气信号 (成交量周期性变化)
        3. 共振增强 (动量与周期的协同)

    参数:
        close: 收盘价 DataFrame
        volume: 成交量 DataFrame
        weriod: 共振评估周期 (默认60, 较长周期)
        window: 最终平滑窗口 (默认20)
        ewm: 是否使用指数加权

    返回:
        成长周期共振因子值 (正值表示成长周期共振信号)
    """

    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算收益率
    returns = close.pct_change()

    # 成长动量信号 (长期上涨趋势)
    growth_momentum = roller_mean(returns, weriod, weriod, method)
    growth_trend = growth_momentum.where(growth_momentum > 0, 0)  # 只关注正增长

    # 周期景气信号 (成交量周期性变化)
    volume_cycle = roller_mean(volume, weriod, weriod, method)
    volume_trend = volume_cycle / roller_mean(volume_cycle, weriod*2, weriod*2, method) - 1
    volume_trend = volume_trend.where(volume_trend > 0, 0)  # 只关注上升周期

    # 共振增强 (动量与周期的协同)
    momentum_cycle_corr = roller_corr(growth_momentum, volume_trend, weriod, weriod, method)
    synergy_signal = growth_trend * volume_trend * (1 + momentum_cycle_corr)

    # 成长周期共振因子
    resonance_factor = synergy_signal

    # 标准化处理
    factor_values = resonance_factor
    factor_mean = roller_mean(factor_values, weriod, weriod, method)
    factor_std = roller_std(factor_values, weriod, weriod, method)
    factor_values = (factor_values - factor_mean) / (factor_std + 1e-8)

    # 最终平滑 (window参数仅用于最终平滑, min_periods=1)
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

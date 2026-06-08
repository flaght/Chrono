"""
kx022 - 分析师预期因子 (近似实现)

研报来源: 真实超预期系列研究之三：从低预期里寻找超预期.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于历史收益模式的超预期识别，通过收益反转和异常表现识别低预期高回报机会
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx022(close, volume, weriod, window, ewm):
    """
    分析师预期因子 (kx022) - 近似实现

    基于从低预期里寻找超预期的策略，通过历史表现模式识别超预期机会。

    核心逻辑:
        1. 计算收益反转模式 (近似预期修正)
        2. 识别异常表现机会 (近似超预期事件)
        3. 构建预期择时信号
        4. 生成分析师预期因子

    因子原理:
        预期因子 = 收益反转 × 异常强度 × 持续性确认
        基于历史模式识别的超预期策略

    参数说明:
    close: 资产收盘价 DataFrame
    volume: 资产成交量 DataFrame
    weriod: 预期计算周期 (默认20)
    window: 确认窗口 (默认20)
    ewm: 是否使用指数加权

    返回值:
        分析师预期因子值
    """
    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 确保数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算价格收益率
    returns = close.pct_change()

    # 收益反转模式识别 (近似预期修正)
    returns_ma = roller_mean(returns, weriod, weriod, method)
    returns_reversal = -1 * (returns - returns_ma)  # 反转信号

    # 异常表现识别 (近似超预期强度)
    returns_std = roller_std(returns, weriod, weriod, method)
    abnormal_returns = returns_reversal / (returns_std + 1e-8)  # 标准化异常收益

    # 成交量配合 (预期修正时的成交量放大)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_ratio = volume / (volume_ma + 1e-8)
    volume_confirmation = (volume_ratio - 1).clip(lower=0)

    # 预期择时信号 (基于个股异常表现)
    abnormal_strength = abnormal_returns.abs()  # 个股异常表现强度
    volume_strength = volume_confirmation.abs()  # 个股成交量确认强度

    expectation_signal = abnormal_strength * volume_strength

    # 预期持续性确认
    expectation_persistence = roller_mean(expectation_signal, weriod, weriod, method)

    # 分析师预期因子
    factor_values = expectation_signal * expectation_persistence


    # 应用最终平滑
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

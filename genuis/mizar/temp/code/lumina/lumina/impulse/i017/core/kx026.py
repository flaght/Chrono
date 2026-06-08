"""
kx026 - VIX情绪择时因子 (近似实现)

研报来源: 指增中性专题报告（一）：基于情绪指标VIX的择时策略.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于价格波动率的VIX情绪近似，用市场波动率作为投资者情绪指标
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx026(close, volume, weriod, window, ewm):
    """
    VIX情绪择时因子 (kx026) - 近似实现

    基于VIX情绪指标的择时策略，通过市场波动率近似投资者情绪状态。

    核心逻辑:
        1. 计算市场波动率 (VIX近似)
        2. 识别情绪极值点
        3. 构建择时信号
        4. 生成情绪因子

    因子原理:
        VIX择时 = 波动率情绪 × 成交量确认 × 反转信号
        基于投资者情绪的择时策略

    参数说明:
    close: 资产收盘价 DataFrame
    volume: 资产成交量 DataFrame
    weriod: 情绪计算周期 (默认20)
    window: 择时确认窗口 (默认20)
    ewm: 是否使用指数加权

    返回值:
        VIX情绪择时因子值
    """
    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 确保数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算收益率
    returns = close.pct_change()

    # 计算波动率 (VIX情绪近似)
    volatility = roller_std(returns, weriod, weriod, method)

    # 波动率标准化 (情绪强度)
    vol_mean = roller_mean(volatility, weriod*2, weriod, method)
    vol_std = roller_std(volatility, weriod*2, weriod, method)
    vix_emotion = (volatility - vol_mean) / (vol_std + 1e-8)

    # 成交量放大 (情绪确认)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_surge = volume / (volume_ma + 1e-8)

    # 情绪择时信号 (高波动率反转)
    emotion_signal = vix_emotion * volume_surge

    # 择时因子 (情绪反转策略)
    factor_values = -emotion_signal  # 负号表示在高波动时做空

    # 应用最终平滑
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

"""
kx002 - 分析师预期调整事件增强因子

研报来源: 因子深度研究系列：分析师预期调整事件增强选股策略全攻略.pdf
实现状态: generated
数据字段: close, volume
"""
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx002(close, volume, weriod, window, ewm):
    """
    分析师预期调整事件增强因子 (kx002)

    基于分析师预期调整事件的增强选股策略，通过事件驱动的价格反应和成交量放大来识别预期调整机会。

    核心逻辑:
        1. 识别预期调整事件
        2. 衡量事件后的价格反应强度
        3. 结合成交量放大效应
        4. 生成事件增强因子

    因子原理:
        预期调整事件增强 = 事件反应强度 × 成交量放大系数
        捕捉分析师预期调整带来的交易机会

    参数说明:
    close: 资产收盘价 DataFrame
    volume: 资产成交量 DataFrame
    weriod: 事件识别周期 (默认20)
    window: 反应强度计算窗口 (默认20)
    ewm: 是否使用指数加权

    返回值:
        分析师预期调整事件增强因子值
    """
    method = 'ewm' if ewm else 'rolling'
    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 确保数据对齐
    #close, volume = close.align(volume, join='inner')

    # 计算价格收益率和波动性
    returns = close.pct_change()
    returns_std = roller_std(returns, weriod, weriod, method)

    # 识别预期调整事件 (价格异动)
    price_change = returns.abs()
    event_threshold = returns_std * 2  # 两倍标准差作为事件阈值
    adjustment_events = (price_change > event_threshold).astype(int)

    # 计算事件反应强度
    event_reaction = roller_sum(adjustment_events, weriod, weriod, method)

    # 成交量放大系数
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_amplification = volume / (volume_ma + 1e-8)

    # 分析师预期调整事件增强因子
    factor_values = event_reaction * volume_amplification

    # 应用最终平滑
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

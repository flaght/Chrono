"""
kx005 - 商品期货市场的趋势因子

研报来源: 因子与指数投资揭秘系列三：商品期货市场的趋势因子.pdf
实现状态: generated
数据字段: close, volume
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx005(close, volume, fast, slow, weriod, window, ewm):
    """
    商品期货市场的趋势因子 (kx005)

    基于商品期货市场的趋势跟踪策略，通过价格趋势强度和成交量确认来识别期货市场趋势。

    核心逻辑:
        1. 计算价格趋势强度
        2. 衡量趋势持续性
        3. 结合成交量确认趋势有效性
        4. 生成期货趋势因子

    因子原理:
        期货趋势因子 = 趋势强度 × 持续性权重 × 成交量确认
        反映商品期货市场的趋势特征

    参数说明:
    close: 资产收盘价 DataFrame
    volume: 资产成交量 DataFrame
    weriod: 趋势计算周期 (默认20)
    window: 趋势确认窗口 (默认20)
    ewm: 是否使用指数加权

    返回值:
        商品期货趋势因子值
    """
    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 确保数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算价格趋势强度 (移动平均斜率)
    price_ma_short = roller_mean(close, fast, fast, method)
    price_ma_long = roller_mean(close, slow, slow, method)
    trend_strength = (price_ma_short - price_ma_long) / (price_ma_long + 1e-8)

    # 计算趋势持续性 (趋势方向的一致性)
    returns = close.pct_change()
    trend_direction = (returns > 0).astype(int) - (returns < 0).astype(int)  # 1, -1, 0
    trend_persistence = roller_mean(trend_direction, weriod, weriod, method)

    # 成交量确认趋势 (趋势强度与成交量的配合)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_ratio = volume / (volume_ma + 1e-8)
    volume_confirmation = roller_mean(volume_ratio, weriod, weriod, method)

    # 商品期货趋势因子
    factor_values = trend_strength * trend_persistence * volume_confirmation

    # 应用最终平滑
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

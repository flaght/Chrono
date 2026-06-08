"""
kx049 - 高频量价因子

研报来源: 量化研究系列报告之十九：破解Alpha投资困境，因子择时方案再探索.pdf
实现状态: generated
数据字段: close, volume
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx049(close, volume, weriod, window, ewm):
    """
    高频量价因子 (kx049)

    基于高频数据的量价关系因子，通过成交量加权的价格变化来衡量量价配合强度。

    核心逻辑:
        1. 计算成交量加权的价格变化
        2. 衡量量价配合强度
        3. 检测成交量异常情况
        4. 生成综合量价因子

    因子原理:
        高频量价因子 = 量价配合强度 × 成交量异常系数
        反映了价格变化与成交量之间的配合关系

    参数说明:
    close: 资产收盘价 DataFrame
    volume: 资产成交量 DataFrame
    weriod: 时间周期参数，用于计算成交量权重 (默认20)
    window: 滚动窗口大小，用于计算配合强度 (默认20)
    ewm: 是否使用指数加权 (默认False)

    返回值:
    高频量价因子值 (数值反映量价配合强度)
    """
    method = 'ewm' if ewm else 'rolling'
    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 确保数据对齐
    #close, volume = close.align(volume, join='inner')

    # 高频量价因子: 成交量加权的价格变化
    vwap_change = close.pct_change()  # 简化为收盘价变化
    volume_sum = roller_mean(volume, weriod, weriod, method) * weriod
    volume_weight = volume / (volume_sum + 1e-8)  # 避免除零

    # 量价配合强度 (使用roller函数替代rolling)
    volume_price_corr = roller_mean(vwap_change * volume_weight, weriod, weriod, method)

    # 成交量异常检测 (使用roller函数)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_ratio = volume / (volume_ma + 1e-8)

    # 综合量价因子
    factor_values = volume_price_corr * volume_ratio

    # 应用最终平滑
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

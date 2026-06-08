"""
kx039 - 流通股本分布因子 (近似实现)

研报来源: 量化技术分析系列之一：利用流通股本分布寻找上涨信号.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于日频价格和成交量数据近似流通股本分布特征，使用价格动量和成交活跃度
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx039(close, volume, weriod, window, ewm):
    """
    流通股本分布因子 (kx039) - 近似实现

    量化技术分析系列，利用流通股本分布寻找上涨信号。
    基于日频数据近似流通股本分布特征。

    近似逻辑:
        1. 价格动量信号 (上涨趋势)
        2. 成交量活跃度 (市场关注度)
        3. 动量稳定性 (持续性评估)

    参数:
        close: 收盘价 DataFrame
        volume: 成交量 DataFrame
        weriod: 分布评估周期 (默认20)
        window: 最终平滑窗口 (默认20)
        ewm: 是否使用指数加权

    返回:
        流通股本分布因子值 (正值表示上涨信号)
    """

    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算收益率
    returns = close.pct_change()

    # 价格动量信号 (上涨趋势强度)
    momentum = roller_mean(returns, weriod, weriod, method)

    # 成交量活跃度 (市场关注度近似)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_ratio = volume / (volume_ma + 1e-8)
    volume_activity = roller_mean(volume_ratio, weriod, weriod, method)

    # 动量稳定性 (趋势持续性)
    momentum_std = roller_std(returns, weriod, weriod, method)
    momentum_stability = momentum / (momentum_std + 1e-8)

    # 价格区间信号 (价格分布特征)
    price_range = (close - roller_min(close, weriod, weriod, 'rolling')) / \
                 (roller_max(close, weriod, weriod, 'rolling') - roller_min(close, weriod, weriod, 'rolling') + 1e-8)
    price_position = roller_mean(price_range, weriod, weriod, method)

    # 流通股本分布因子 = 动量 × 活跃度 × 稳定性 × 价格位置
    distribution_factor = momentum * volume_activity * momentum_stability * price_position

    # 标准化处理
    factor_values = distribution_factor

    # 最终平滑 (window参数仅用于最终平滑, min_periods=1)
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

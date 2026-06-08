"""
kx015 - 基于大单的alpha因子构建 (近似实现)

研报来源: 因子选股系列之七十九：基于大单的alpha因子构建.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于日频成交量激增事件和价格冲击近似大单交易行为，使用成交量异常×价格反应×持续效应
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx015(close, volume, weriod, window, ewm):
    """
    基于大单的alpha因子构建 (kx015) - 近似实现

    基于大单交易行为的alpha因子构建，通过成交量激增事件和价格冲击效应来近似大单交易行为。

    核心逻辑:
        1. 识别成交量异常事件 (近似大单出现)
        2. 衡量价格冲击效应
        3. 评估持续交易效应
        4. 生成大单alpha因子

    因子原理:
        大单alpha因子 = 成交量异常 × 价格冲击 × 持续效应
        基于日频数据近似大单交易行为特征

    参数说明:
    close: 资产收盘价 DataFrame
    volume: 资产成交量 DataFrame
    weriod: 异常检测周期 (默认20)
    window: 效应评估窗口 (默认20)
    ewm: 是否使用指数加权

    返回值:
        大单alpha因子值
    """
    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 确保数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 识别成交量异常事件 (近似大单出现)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_ratio = volume / (volume_ma + 1e-8)
    volume_anomaly = (volume_ratio - 1).clip(lower=0)  # 只关注成交量激增

    # 衡量价格冲击效应
    returns = close.pct_change()
    price_impact = np.abs(returns) * volume_anomaly  # 成交量激增时的价格变化强度

    # 评估持续交易效应 (后续价格变化的持续性)
    future_returns = returns.shift(-1)  # 次日收益率
    persistence_effect = roller_mean(future_returns, weriod, weriod, method) * volume_anomaly

    # 大单alpha因子 (近似实现)
    factor_values = volume_anomaly * price_impact * (persistence_effect + 1)


    # 应用最终平滑
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

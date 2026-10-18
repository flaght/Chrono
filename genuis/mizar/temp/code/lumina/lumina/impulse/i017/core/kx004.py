"""
kx004 - 高频量价选股因子

研报来源: 因子深度研究系列：高频量价选股因子初探.pdf
实现状态: generated
数据字段: close, volume
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx004(close, volume, weriod, window, ewm):
    """
    高频量价选股因子 (kx004)

    基于高频数据的量价关系，通过成交量和价格的协同变化来识别选股机会。

    核心逻辑:
        1. 计算高频价格变化
        2. 分析成交量反应强度
        3. 衡量量价配合度
        4. 生成高频选股信号

    因子原理:
        高频量价因子 = 价格变化强度 × 成交量反应系数
        捕捉高频交易中的量价配合机会

    参数说明:
    close: 资产收盘价 DataFrame
    volume: 资产成交量 DataFrame
    weriod: 高频分析周期 (默认20)
    window: 信号聚合窗口 (默认20)
    ewm: 是否使用指数加权

    返回值:
        高频量价选股因子值
    """
    method = 'ewm' if ewm else 'rolling'
    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 确保数据对齐
    #close, volume = close.align(volume, join='inner')

    # 计算高频价格变化强度
    returns = close.pct_change()
    price_volatility = roller_std(returns, weriod, weriod, method)

    # 成交量反应强度
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_reaction = volume / (volume_ma + 1e-8)

    # 量价配合度 (价格变化与成交量的相关性)
    price_volume_corr = roller_corr(returns, volume_reaction, weriod, weriod, method)

    # 高频量价选股因子
    factor_values = price_volatility * volume_reaction * price_volume_corr

    # 应用最终平滑
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

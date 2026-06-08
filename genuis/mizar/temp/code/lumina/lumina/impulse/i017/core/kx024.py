"""
kx024 - 拥挤度行业轮动因子 (近似实现)

研报来源: 资产配置研究系列之四：基于拥挤度判断的行业轮动策略.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于成交量和价格波动的拥挤度判断，实现行业轮动策略
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx024(close, volume, weriod, window, ewm):
    """
    拥挤度行业轮动因子 (kx024) - 近似实现

    基于拥挤度判断的行业轮动策略，通过成交量和价格变动识别市场拥挤状态。

    核心逻辑:
        1. 计算成交量拥挤度指标
        2. 识别价格拥挤状态
        3. 构建轮动信号
        4. 生成拥挤度因子

    因子原理:
        拥挤度轮动 = 成交量拥挤 × 价格拥挤 × 轮动强度
        基于市场拥挤度的行业轮动策略

    参数说明:
    close: 资产收盘价 DataFrame
    volume: 资产成交量 DataFrame
    weriod: 拥挤度计算周期 (默认20)
    window: 轮动确认窗口 (默认20)
    ewm: 是否使用指数加权

    返回值:
        拥挤度行业轮动因子值
    """
    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 确保数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算收益率
    returns = close.pct_change()

    # 成交量拥挤度 (相对成交量强度)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_crowding = volume / (volume_ma + 1e-8)

    # 价格拥挤度 (价格波动放大)
    returns_volatility = roller_std(returns, weriod, weriod, method)
    returns_ma = roller_mean(returns.abs(), weriod, weriod, method)
    price_crowding = returns_volatility / (returns_ma + 1e-8)

    # 拥挤度综合指标
    crowding_index = volume_crowding * price_crowding

    # 轮动信号 (拥挤度反转)
    crowding_ma = roller_mean(crowding_index, weriod, weriod, method)
    rotation_signal = crowding_ma - crowding_index

    # 拥挤度行业轮动因子
    factor_values = rotation_signal


    # 应用最终平滑
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

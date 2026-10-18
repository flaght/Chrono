"""
kx036 - 成交量激增因子 (近似实现)

研报来源: 多因子选股系列研究之一：成交量激增时刻蕴含的alpha信息.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于日频数据近似成交量激增事件，使用成交量异常放大×价格响应
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx036(close, volume, weriod, window, ewm):
    """
    成交量激增因子 (kx036) - 近似实现

    成交量激增时刻蕴含的alpha信息。
    基于日频数据近似成交量激增事件。

    近似逻辑:
        1. 识别成交量激增事件 (成交量异常放大)
        2. 评估价格响应强度 (激增后的价格变动)
        3. 构建成交量激增因子

    参数:
        close: 收盘价 DataFrame
        volume: 成交量 DataFrame
        weriod: 激增评估周期 (默认20)
        window: 最终平滑窗口 (默认20)
        ewm: 是否使用指数加权

    返回:
        成交量激增因子值 (正值表示成交量激增事件)
    """

    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算收益率
    returns = close.pct_change()

    # 成交量激增检测 (相对历史平均水平的放大)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_ratio = volume / (volume_ma + 1e-8)

    # 激增阈值判断 (基于个股历史统计)
    volume_ratio_mean = roller_mean(volume_ratio, weriod, weriod, method)
    volume_ratio_std = roller_std(volume_ratio, weriod, weriod, method)
    #volume_surge = np.where(volume_ratio > volume_ratio_mean + volume_ratio_std, 1, 0)
    volume_surge = (volume_ratio > volume_ratio_mean + volume_ratio_std).astype(int)

    # 价格响应强度 (激增后的价格变动幅度)
    returns_abs = np.abs(returns)

    # 激增后价格响应 (成交量激增当日的价格变动)
    price_response = returns_abs * volume_surge

    # 后续价格动量 (激增后几日的价格趋势)
    momentum_short = roller_mean(returns, 5, 3, method)
    momentum_response = momentum_short * volume_surge

    # 成交量激增因子 = 激增强度 × 价格响应 × 后续动量
    volume_surge_factor = volume_ratio * price_response * (momentum_response + 1)

    # 标准化处理
    factor_values = volume_surge_factor

    # 最终平滑 (window参数仅用于最终平滑, min_periods=1)
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

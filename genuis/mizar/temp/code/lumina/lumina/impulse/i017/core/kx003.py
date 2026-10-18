"""
kx003 - 多层次订单失衡及订单斜率因子

研报来源: 因子深度研究系列：多层次订单失衡及订单斜率因子.pdf
实现状态: generated
数据字段: close, volume
"""

import pandas as pd
import numpy as np

from lumina.impulse.fixed import *

def kx003(close, volume, weriod, window, ewm):
    """
    多层次订单失衡及订单斜率因子 (kx003)

    基于订单簿多层次信息的失衡状态，通过价格和成交量的关系来衡量订单失衡程度和斜率变化。

    核心逻辑:
        1. 计算订单失衡指标
        2. 衡量订单簿斜率变化
        3. 结合多层次价格信息
        4. 生成订单失衡因子

    因子原理:
        订单失衡因子 = 失衡强度 × 斜率变化系数
        反映了订单簿多层次的供需失衡状态

    参数说明:
    close: 资产收盘价 DataFrame
    volume: 资产成交量 DataFrame
    weriod: 失衡计算周期 (默认20)
    window: 斜率计算窗口 (默认20)
    ewm: 是否使用指数加权

    返回值:
        多层次订单失衡及订单斜率因子值
    """
    method = 'ewm' if ewm else 'rolling'
    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 确保数据对齐
    #close, volume = close.align(volume, join='inner')

    # 计算价格变化和成交量变化
    returns = close.pct_change()
    volume_change = volume.pct_change()

    # 订单失衡指标 (价格变化与成交量变化的比值)
    imbalance_ratio = returns / (volume_change.abs() + 1e-8)

    # 多层次订单失衡强度
    imbalance_strength = roller_mean(imbalance_ratio.abs(), weriod, weriod, method)

    # 订单斜率变化 (失衡指标的变化趋势)
    imbalance_trend = imbalance_ratio - roller_mean(imbalance_ratio, weriod, weriod, method)
    slope_change = roller_std(imbalance_trend, weriod, weriod, method)

    # 多层次订单失衡及订单斜率因子
    factor_values = imbalance_strength * slope_change

    # 标准化处理
    factor_values = (factor_values - roller_mean(factor_values, weriod, weriod, method)) / \
                   (roller_std(factor_values, weriod, weriod, method) + 1e-8)

    # 应用最终平滑
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

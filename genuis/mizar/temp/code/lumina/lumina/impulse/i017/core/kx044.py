"""
kx044 - 大消费板块轮动因子 (近似实现)

研报来源: 开源量化评论（31）：大消费板块的轮动与选股.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于日频数据近似大消费板块轮动因子，使用价格稳定性和成交特征
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx044(close, volume, weriod, window, ewm):
    """
    大消费板块轮动因子 (kx044) - 近似实现

    开源量化评论（31）：大消费板块的轮动与选股。
    基于日频数据近似大消费板块轮动逻辑。

    近似逻辑:
        1. 价格稳定性信号 (消费股防御性特征)
        2. 成交量温和增长 (稳定需求特征)
        3. 轮动时机识别 (相对强度)

    参数:
        close: 收盘价 DataFrame
        volume: 成交量 DataFrame
        weriod: 轮动评估周期 (默认60, 较长周期)
        window: 最终平滑窗口 (默认20)
        ewm: 是否使用指数加权

    返回:
        大消费轮动因子值 (正值表示消费板块轮动机会)
    """

    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算收益率
    returns = close.pct_change()

    # 价格稳定性信号 (消费股特征 - 相对低波动)
    price_volatility = roller_std(returns, weriod, weriod, method)
    stability_score = 1 / (price_volatility + 1e-8)  # 稳定性得分

    # 成交量温和增长 (稳定需求特征)
    volume_growth = (volume / volume.shift(weriod) - 1).fillna(0)
    #volume_stability = np.where(volume_growth > 0, volume_growth, 0)  # 只关注正增长
    volume_stability = volume_growth.where(volume_growth > 0, 0)
    volume_stability = roller_mean(volume_stability, weriod, weriod, method)

    # 轮动时机识别 (相对强度 - 抗跌能力)
    price_strength = returns - roller_mean(returns, weriod, weriod, method)
    #relative_strength = np.where(price_strength > 0, price_strength, 0)  # 只关注相对强势
    relative_strength = price_strength.where(price_strength > 0, 0)

    # 大消费轮动因子 = 稳定性 × 成交温和 × 相对强度
    rotation_factor = stability_score * volume_stability * relative_strength

    # 标准化处理
    factor_values = rotation_factor

    # 最终平滑 (window参数仅用于最终平滑, min_periods=1)
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

"""
kx042 - TMT行业量化选股因子 (近似实现)

研报来源: 开源证券量化评论（6）：TMT行业的量化选股方案.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于日频数据近似TMT行业选股因子，使用技术指标和成交活跃度
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx042(close, volume, weriod, window, ewm):
    """
    TMT行业量化选股因子 (kx042) - 近似实现

    开源证券量化评论（6）：TMT行业的量化选股方案。
    基于日频数据近似TMT行业选股逻辑。

    近似逻辑:
        1. 技术成长信号 (价格动量和技术突破)
        2. 成交活跃度 (市场关注度)
        3. 波动稳定性 (成长股特征)

    参数:
        close: 收盘价 DataFrame
        volume: 成交量 DataFrame
        weriod: 选股评估周期 (默认20)
        window: 最终平滑窗口 (默认20)
        ewm: 是否使用指数加权

    返回:
        TMT选股因子值 (正值表示更具TMT选股潜力)
    """

    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算收益率
    returns = close.pct_change()

    # 技术成长信号 (动量+突破)
    momentum = roller_mean(returns, weriod, weriod, method)
    price_trend = np.where(close > roller_max(close, weriod, weriod, 'rolling'), 1, 0)
    growth_signal = momentum + price_trend

    # 成交活跃度 (TMT关注度)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_ratio = volume / (volume_ma + 1e-8)
    activity_score = roller_mean(volume_ratio, weriod, weriod, method)

    # 波动稳定性 (成长股特征 - 高波动但有趋势)
    volatility = roller_std(returns, weriod, weriod, method)
    trend_strength = np.abs(momentum)
    stability_score = volatility / (trend_strength + 1e-8)  # 波动相对趋势强度

    # TMT选股因子 = 成长信号 × 活跃度 × (1/波动稳定性)
    tmt_factor = growth_signal * activity_score * (1 / (stability_score + 1e-8))

    # 标准化处理
    factor_values = tmt_factor

    # 最终平滑 (window参数仅用于最终平滑, min_periods=1)
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

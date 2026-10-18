"""
kx046 - 分析师预期调整因子 (近似实现)

研报来源: 开源量化评论（99）：深度学习赋能分析师行为，更稳的盈利预期调整组合.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于日频数据近似分析师预期调整因子，使用价格修正和成交量变化
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx046(close, volume, weriod, window, ewm):
    """
    分析师预期调整因子 (kx046) - 近似实现

    开源量化评论（99）：深度学习赋能分析师行为，更稳的盈利预期调整组合。
    基于日频数据近似分析师预期调整逻辑。

    近似逻辑:
        1. 价格修正信号 (预期调整迹象)
        2. 成交量异常变化 (预期变化确认)
        3. 预期稳定性评估

    参数:
        close: 收盘价 DataFrame
        volume: 成交量 DataFrame
        weriod: 预期评估周期 (默认30)
        window: 最终平滑窗口 (默认20)
        ewm: 是否使用指数加权

    返回:
        预期调整因子值 (正值表示预期向上调整机会)
    """

    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算收益率
    returns = close.pct_change()

    # 价格修正信号 (预期调整迹象)
    price_reversal = returns - roller_mean(returns, weriod, weriod, method)
    #revision_signal = np.where(price_reversal > 0, price_reversal, 0)  # 只关注向上修正
    revision_signal = price_reversal.where(price_reversal > 0, 0)

    # 成交量异常变化 (预期变化确认)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_spike = volume / (volume_ma + 1e-8)

    # 基于个股历史统计的成交量异常 (单品种逻辑)
    volume_spike_mean = roller_mean(volume_spike, weriod, weriod, method)
    volume_spike_std = roller_std(volume_spike, weriod, weriod, method)
    volume_anomaly = np.where(volume_spike > volume_spike_mean + volume_spike_std, volume_spike, 0)

    # 预期稳定性评估 (一致性调整信号)
    price_stability = 1 / (roller_std(returns, weriod, weriod, method) + 1e-8)
    volume_stability = 1 / (roller_std(volume_spike, weriod, weriod, method) + 1e-8)
    consistency_score = price_stability * volume_stability

    # 预期调整因子 = 修正信号 × 成交异常 × 一致性
    expectation_factor = revision_signal * volume_anomaly * consistency_score

    # 标准化处理
    factor_values = expectation_factor

    # 最终平滑 (window参数仅用于最终平滑, min_periods=1)
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

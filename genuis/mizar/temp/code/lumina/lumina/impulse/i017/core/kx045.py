"""
kx045 - 行业泡沫拥挤度因子 (近似实现)

研报来源: 开源量化评论（45）：行业泡沫膨胀与破裂的识别，以拥挤之名.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于日频数据近似行业泡沫拥挤度因子，使用成交量集中度和价格异常
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx045(close, volume, weriod, window, ewm):
    """
    行业泡沫拥挤度因子 (kx045) - 近似实现

    开源量化评论（45）：行业泡沫膨胀与破裂的识别，以拥挤之名。
    基于日频数据近似泡沫拥挤度识别。

    近似逻辑:
        1. 成交量集中度 (资金拥挤信号)
        2. 价格异常波动 (泡沫迹象)
        3. 拥挤度综合评分

    参数:
        close: 收盘价 DataFrame
        volume: 成交量 DataFrame
        weriod: 拥挤度评估周期 (默认20)
        window: 最终平滑窗口 (默认20)
        ewm: 是否使用指数加权

    返回:
        泡沫拥挤度因子值 (正值表示泡沫/拥挤风险)
    """

    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算收益率
    returns = close.pct_change()

    # 成交量集中度 (资金拥挤信号)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_deviation = volume / (volume_ma + 1e-8) - 1
    #volume_concentration = np.where(volume_deviation > 0, volume_deviation, 0)  # 只关注高成交
    volume_concentration = volume_deviation.where(volume_deviation > 0, 0)

    # 价格异常波动 (泡沫迹象)
    returns_volatility = roller_std(returns, weriod, weriod, method)
    returns_mean = roller_mean(returns, weriod, weriod, method)
    price_anomaly = np.abs(returns - returns_mean) / (returns_volatility + 1e-8)

    # 价格趋势强度 (泡沫膨胀迹象)
    trend_momentum = roller_mean(returns, weriod, weriod, method)
    trend_acceleration = trend_momentum - roller_mean(trend_momentum, weriod, weriod, method)

    # 泡沫拥挤度因子 = 成交集中 × 价格异常 × 趋势加速
    crowding_factor = volume_concentration * price_anomaly * trend_acceleration

    # 标准化处理
    factor_values = crowding_factor

    # 最终平滑 (window参数仅用于最终平滑, min_periods=1)
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

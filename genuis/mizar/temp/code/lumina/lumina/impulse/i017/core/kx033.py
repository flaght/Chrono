"""
kx033 - 股债相关性驱动因子 (近似实现)

研报来源: 学界纵横系列之四十六：股债相关性驱动因素研究.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于股票波动率和趋势特征近似股债相关性，使用波动率相关×趋势背离×风险溢价信号
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx033(close, volume, fast, slow, weriod, window, ewm):
    """
    股债相关性驱动因子 (kx033) - 近似实现

    股债相关性驱动因素研究。
    基于股票波动率和趋势特征构建股债相关性驱动因子。

    近似逻辑:
        1. 计算股票波动率特征 (近似市场风险水平)
        2. 识别趋势背离信号 (近似风险偏好变化)
        3. 构建风险溢价信号 (近似股债配置信号)

    参数:
        close: 收盘价 DataFrame
        volume: 成交量 DataFrame
        weriod: 相关性评估周期 (默认20)
        window: 最终平滑窗口 (默认20)
        ewm: 是否使用指数加权

    返回:
        股债相关性驱动因子值 (正值表示股债相关性增强)
    """

    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算收益率
    returns = close.pct_change()

    # 股票波动率特征 (市场风险水平)
    stock_volatility = roller_std(returns, weriod, weriod, method)

    # 趋势信号 (市场方向)
    trend_short = roller_mean(returns, fast, fast, method)
    trend_long = roller_mean(returns, slow, slow, method)
    trend_divergence = trend_short - trend_long  # 趋势背离

    # 成交量确认 (市场活跃度)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_ratio = volume / (volume_ma + 1e-8)

    # 风险溢价信号 (波动率×成交量调整)
    risk_premium = stock_volatility * volume_ratio

    # 股债相关性驱动因子 (单品种时序逻辑)
    # 核心逻辑：基于个股历史数据判断相关性驱动信号
    # 波动率压力：波动率高于个股历史均值+1倍标准差
    # 趋势冲突：趋势背离绝对值高于个股历史均值+1倍标准差
    # 溢价信号：风险溢价高于个股历史均值

    # 计算个股历史统计 (单品种时序)
    vol_mean = roller_mean(stock_volatility, weriod, weriod, method)
    vol_std = roller_std(stock_volatility, weriod, weriod, method)
    trend_mean = roller_mean(np.abs(trend_divergence), weriod, weriod, method)
    trend_std = roller_std(np.abs(trend_divergence), weriod, weriod, method)
    premium_mean = roller_mean(risk_premium, weriod, weriod, method)

    # 基于个股历史判断相关性驱动 (连续值逻辑)
    # 将离散条件改为连续值，提供更丰富的信息

    # 波动率压力：标准化后的波动率信号 (正值表示波动率压力大)
    volatility_zscore = (stock_volatility - vol_mean) / (vol_std + 1e-8)
    volatility_stress = volatility_zscore.clip(-3, 3)  # 限制在-3到3之间

    # 趋势冲突：标准化后的趋势背离信号 (正值表示趋势冲突大)
    trend_abs = np.abs(trend_divergence)
    trend_zscore = (trend_abs - trend_mean) / (trend_std + 1e-8)
    trend_conflict = trend_zscore.clip(-3, 3)

    # 溢价信号：标准化后的风险溢价信号 (正值表示溢价水平高)
    premium_zscore = (risk_premium - premium_mean) / (roller_std(risk_premium, weriod, weriod, method) + 1e-8)
    premium_signal = premium_zscore.clip(-3, 3)

    # 相关性驱动得分 = 波动压力 × 趋势冲突 × 溢价信号 (连续值相乘)
    correlation_driver = volatility_stress * trend_conflict * premium_signal

    # 标准化处理
    factor_values = correlation_driver.astype(float)
    # 最终平滑 (window参数仅用于最终平滑, min_periods=1)
    factor_values = roller_mean(factor_values, window, window, method)

    return factor_values

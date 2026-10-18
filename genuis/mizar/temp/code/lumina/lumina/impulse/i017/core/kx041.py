"""
kx041 - 市场分配建底仓因子 (近似实现)

研报来源: 量化视角看市场：绝对收益（三），沪深300等市场分配建底仓策略.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于日频数据近似市场分配建底仓策略，使用价格趋势和风险评估
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx041(close, volume, weriod, window, ewm):
    """
    市场分配建底仓因子 (kx041) - 近似实现

    量化视角看市场，绝对收益系列，沪深300等市场分配建底仓策略。
    基于日频数据近似市场配置和建仓时机。

    近似逻辑:
        1. 市场趋势评估 (上涨/下跌周期)
        2. 风险调整收益 (夏普比率近似)
        3. 建仓时机信号 (底部识别)

    参数:
        close: 收盘价 DataFrame
        volume: 成交量 DataFrame
        weriod: 市场评估周期 (默认60)
        window: 最终平滑窗口 (默认20)
        ewm: 是否使用指数加权

    返回:
        市场分配建底仓因子值 (正值表示建仓机会)
    """

    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算收益率
    returns = close.pct_change()

    # 市场趋势评估 (长期趋势方向)
    trend_momentum = roller_mean(returns, weriod, weriod, method)
    trend_strength = np.abs(trend_momentum)  # 趋势强度

    # 风险调整收益 (夏普比率近似)
    return_volatility = roller_std(returns, weriod, weriod, method)
    risk_adjusted_return = trend_momentum / (return_volatility + 1e-8)

    # 成交量确认 (市场参与度)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_ratio = volume / (volume_ma + 1e-8)
    market_participation = roller_mean(volume_ratio, weriod, weriod, method)

    # 建仓时机信号 (底部识别 - 反转信号)
    price_position = (close - roller_min(close, weriod, weriod, 'rolling')) / \
                    (roller_max(close, weriod, weriod, 'rolling') - roller_min(close, weriod, weriod, 'rolling') + 1e-8)

    # 组合建仓因子 = 风险调整收益 × 市场参与度 × (1-价格位置)
    # 价格位置越低(接近底部)，建仓信号越强
    allocation_factor = risk_adjusted_return * market_participation * (1 - price_position)

    # 标准化处理
    factor_values = allocation_factor
    #factor_mean = roller_mean(factor_values, weriod, 30, method)
    #factor_std = roller_std(factor_values, weriod, 30, 'rolling')
    #factor_values = (factor_values - factor_mean) / (factor_std + 1e-8)

    # 最终平滑 (window参数仅用于最终平滑, min_periods=1)
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

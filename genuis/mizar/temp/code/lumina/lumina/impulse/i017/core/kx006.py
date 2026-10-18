"""
kx006 - 单商品指数编制概述及优化因子

研报来源: 因子与指数投资揭秘系列十二：单商品指数编制概述及优化.pdf
实现状态: generated
数据字段: close, volume
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx006(close, volume, weriod, window, ewm):
    """
    单商品指数编制概述及优化因子 (kx006)

    基于单商品指数编制方法的优化策略，通过价格权重调整和波动率控制来构建优化的商品指数因子。

    核心逻辑:
        1. 计算价格权重调整因子
        2. 衡量波动率控制指标
        3. 结合成交量权重优化
        4. 生成指数编制优化因子

    因子原理:
        指数优化因子 = 价格权重 × 波动率控制 × 成交量权重
        反映商品指数编制的最优权重分配

    参数说明:
    close: 资产收盘价 DataFrame
    volume: 资产成交量 DataFrame
    weriod: 权重计算周期 (默认20)
    window: 优化确认窗口 (默认20)
    ewm: 是否使用指数加权

    返回值:
        单商品指数编制优化因子值
    """
    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 确保数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算价格权重调整 (基于价格变化率的权重)
    returns = close.pct_change()
    price_weight = roller_mean(np.abs(returns), weriod, weriod, method)
    price_weight = 1 / (price_weight + 1e-8)  # 波动率越小权重越大

    # 计算波动率控制指标 (价格标准差的逆)
    volatility_control = 1 / (roller_std(returns, weriod, weriod, method) + 1e-8)

    # 成交量权重优化 (成交量稳定性权重)
    volume_stability = roller_std(volume, weriod, weriod, method) / (roller_mean(volume, weriod, weriod, method) + 1e-8)
    volume_weight = 1 / (volume_stability + 1e-8)  # 成交量越稳定权重越大

    # 单商品指数编制优化因子
    factor_values = price_weight * volatility_control * volume_weight


    # 应用最终平滑
    factor_values = roller_mean(factor_values, window, window, method)

    return factor_values

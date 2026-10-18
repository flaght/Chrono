"""
kx027 - 资金流选股因子 (近似实现)

研报来源: 资金流选股因子：主力资金杠杆效率.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于成交量和价格变动的资金流近似，用成交量加权价格变动作为资金流度量
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx027(close, volume, weriod, window, ewm):
    """
    资金流选股因子 (kx027) - 近似实现

    基于资金流的选股因子，通过成交量和价格变动度量主力资金效率。

    核心逻辑:
        1. 计算资金流量指标
        2. 评估资金效率
        3. 构建选股信号
        4. 生成资金流因子

    因子原理:
        资金流因子 = 资金流量 × 资金效率 × 持续性权重
        基于主力资金动向的选股策略

    参数说明:
    close: 资产收盘价 DataFrame
    volume: 资产成交量 DataFrame
    weriod: 资金流计算周期 (默认20)
    window: 选股确认窗口 (默认20)
    ewm: 是否使用指数加权

    返回值:
        资金流选股因子值
    """
    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 确保数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算价格变动
    returns = close.pct_change()

    # 资金流量 (成交量加权价格变动)
    money_flow = returns * volume

    # 资金效率 (标准化资金流量)
    mf_ma = roller_mean(money_flow, weriod, weriod, method)
    mf_std = roller_std(money_flow, weriod, weriod, method)
    money_flow_efficiency = (money_flow - mf_ma) / (mf_std + 1e-8)

    # 资金流持续性 (资金流趋势)
    flow_trend = roller_mean(money_flow_efficiency, weriod, weriod, method)

    # 资金流选股因子
    factor_values = money_flow_efficiency * flow_trend

    # 应用最终平滑
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

"""
kx032 - 盈余公告前异常波动因子 (近似实现)

研报来源: 学术文献研究系列第14期：从盈余公告前异常特质波动看上市公司信息泄露风险.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于价格波动异常和成交量放大近似盈余公告前信息泄露效应
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx032(close, volume, weriod, window, ewm):
    """
    盈余公告前异常波动因子 (kx032) - 近似实现

    从盈余公告前异常特质波动看上市公司信息泄露风险。
    基于价格波动异常和成交量放大识别潜在的信息泄露信号。

    近似逻辑:
        1. 计算特质波动率的异常放大 (信息泄露迹象)
        2. 识别成交量异常放大事件 (交易活跃度)
        3. 构建信息泄露风险因子

    参数:
        close: 收盘价 DataFrame
        volume: 成交量 DataFrame
        weriod: 波动评估周期 (默认20)
        window: 最终平滑窗口 (默认20)
        ewm: 是否使用指数加权

    返回:
        信息泄露风险因子值 (正值表示潜在信息泄露风险)
    """

    # 参数验证
    if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
        raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 数据对齐
    close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算收益率
    returns = close.pct_change()

    # 特质波动率 (简化为历史波动率)
    historical_vol = roller_std(returns, weriod, weriod, weriod)
    current_vol = roller_std(returns, weriod, weriod, method)

    # 波动率异常放大 (信息泄露迹象)
    vol_ratio = current_vol / (historical_vol + 1e-8)
    vol_anomaly = np.maximum(vol_ratio - 1, 0)  # 只取放大部分

    # 成交量异常放大
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_ratio = volume / (volume_ma + 1e-8)
    volume_anomaly = np.maximum(volume_ratio - 1, 0)  # 只取放大部分

    # 价格跳跃信号 (异常价格变动)
    price_jump = np.abs(returns) - historical_vol
    price_jump = np.maximum(price_jump, 0)

    # 信息泄露风险因子 = 波动异常 × 成交量异常 × 价格跳跃
    leakage_risk = vol_anomaly * volume_anomaly * (price_jump + 1)

    # 标准化处理
    factor_values = leakage_risk
    # 最终平滑 (window参数仅用于最终平滑, min_periods=1)
    factor_values = roller_mean(factor_values, window, window, method)

    return factor_values

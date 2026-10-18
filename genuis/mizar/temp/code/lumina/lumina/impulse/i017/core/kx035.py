"""
kx035 - 日内分时因子 (近似实现)

研报来源: 多因子系列报告之八：高频因子，日内分时成交量蕴藏玄机.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于日频数据近似日内分时分布，使用成交量集中度×价格路径复杂性
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx035(close, volume, weriod, window, ewm):
    """
    日内分时因子 (kx035) - 近似实现

    高频因子，日内分时成交量蕴藏玄机。
    基于日频数据近似日内分时分布特征。

    近似逻辑:
        1. 计算成交量集中度 (近似日内分布不均匀性)
        2. 计算价格路径复杂性 (近似日内价格波动模式)
        3. 构建日内分时因子

    参数:
        close: 收盘价 DataFrame
        volume: 成交量 DataFrame
        weriod: 分时评估周期 (默认20)
        window: 最终平滑窗口 (默认20)
        ewm: 是否使用指数加权

    返回:
        日内分时因子值 (正值表示日内分布特征明显)
    """

    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算收益率
    returns = close.pct_change()

    # 成交量集中度 (日内成交量分布的集中程度)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_std = roller_std(volume, weriod, weriod, method)
    volume_concentration = volume_std / (volume_ma + 1e-8)  # 成交量波动系数

    # 价格路径复杂性 (日内价格变动的复杂程度)
    returns_vol = roller_std(returns, weriod, weriod, method)
    returns_skew = roller_skew(returns, weriod, weriod, 'rolling')
    returns_kurt = roller_kurt(returns, weriod, weriod, 'rolling')
    #returns_skew = returns.rolling(window=weriod, min_periods=1).skew()  # 偏度
    #returns_kurt = returns.rolling(window=weriod, min_periods=1).kurt()  # 峰度

    # 填充NaN值
    returns_skew = returns_skew.fillna(0)
    returns_kurt = returns_kurt.fillna(3)  # 正态分布峰度为3

    # 路径复杂性指标
    path_complexity = returns_vol * (np.abs(returns_skew) + np.abs(returns_kurt - 3))

    # 日内分时因子 = 成交量集中度 × 价格路径复杂性
    # 反映日内交易行为的复杂性和集中性
    intraday_factor = volume_concentration * path_complexity

    # 标准化处理
    factor_values = intraday_factor

    # 最终平滑 (window参数仅用于最终平滑, min_periods=1)
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

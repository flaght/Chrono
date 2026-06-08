"""
kx037 - 股价跳跃因子 (近似实现)

研报来源: 多因子选股系列研究之六：个股股价跳跃及其对振幅因子的改进.pdf
实现状态: generated_approximate
数据字段: close, volume
近似说明: 基于日频数据近似股价跳跃事件，使用价格异常变动×成交量确认
"""

# 必需的导入
import pandas as pd
import numpy as np
from lumina.impulse.fixed import *

def kx037(close, volume, weriod, window, ewm):
    """
    股价跳跃因子 (kx037) - 近似实现

    个股股价跳跃及其对振幅因子的改进。
    基于日频数据近似股价跳跃事件。

    近似逻辑:
        1. 检测价格跳跃事件 (异常价格变动)
        2. 评估成交量确认 (跳跃的成交量支持)
        3. 构建股价跳跃因子

    参数:
        close: 收盘价 DataFrame
        volume: 成交量 DataFrame
        weriod: 跳跃评估周期 (默认20)
        window: 最终平滑窗口 (默认20)
        ewm: 是否使用指数加权

    返回:
        股价跳跃因子值 (正值表示股价跳跃事件)
    """

    # 参数验证
    #if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
    #    raise ValueError("close 和 volume 必须是 pandas DataFrame")

    # 数据对齐
    #close, volume = close.align(volume, join='inner')

    method = 'ewm' if ewm else 'rolling'

    # 计算收益率
    returns = close.pct_change()

    # 历史波动率 (基准波动水平)
    historical_vol = roller_std(returns, weriod, weriod, method)

    # 价格跳跃检测 (当日收益率超过历史波动率的倍数)
    returns_abs = np.abs(returns)
    vol_threshold = historical_vol * 2  # 2倍历史波动率为跳跃阈值
    price_jump = (returns_abs - vol_threshold).where(returns_abs > vol_threshold, 0)

    # 成交量确认 (跳跃当日的成交量放大)
    volume_ma = roller_mean(volume, weriod, weriod, method)
    volume_ratio = volume / (volume_ma + 1e-8)

    # 基于个股历史统计的成交量确认
    volume_ratio_mean = roller_mean(volume_ratio, weriod, weriod, method)
    volume_ratio_std = roller_std(volume_ratio, weriod, weriod, method)
    volume_confirmation = (volume_ratio > volume_ratio_mean + volume_ratio_std * 0.5).astype(int)

    # 跳跃方向性 (上涨跳跃 vs 下跌跳跃)
    jump_direction = (returns > 0).astype(int).replace(0, -1)  # 正向跳跃更显著

    # 股价跳跃因子 = 跳跃幅度 × 成交量确认 × 方向权重
    jump_factor = price_jump * volume_confirmation * (jump_direction + 1)

    # 标准化处理
    factor_values = jump_factor

    # 最终平滑 (window参数仅用于最终平滑, min_periods=1)
    factor_values = roller_mean(factor_values, window, 1, method)

    return factor_values

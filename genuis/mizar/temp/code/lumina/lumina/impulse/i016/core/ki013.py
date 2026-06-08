# -*- encoding:utf-8 -*-
"""
gs001: 高频波动率因子 (minute_return_volatility)

来源: 国盛证券 - "薪火"量化分析系列研究（三）：红利低波的增强方案

原始逻辑:
1. 每个交易日，计算分钟涨跌幅的标准差 → vol_daily
2. 回看20个交易日，计算 vol_daily 的变异系数 (std/mean)
3. 变异系数越小，波动越稳定

期货适配:
- 使用滚动窗口计算日内波动率
- 计算波动率的变异系数作为稳定性指标
- 低值表示波动稳定，高值表示波动不稳定
"""
import numpy as np
from lumina.impulse.fixed import *


def ki013(close, window, weriod, ewm=False):
    """
    高频波动率因子

    计算波动率的变异系数 (coefficient of variation)
    CV = std(vol) / mean(vol)
    CV 越小表示波动越稳定

    参数:
        close: 收盘价 (1分钟数据)
        window: 外层平滑窗口
        weriod: 计算周期 (建议240=1天)
        ewm: 是否使用指数加权

    返回:
        alpha: 负的变异系数 (低值表示高波动/不稳定，用于做空)
    """
    method = 'ewm' if ewm else 'rolling'

    # 计算分钟收益率
    rets = safe_log(close)

    # 计算滚动波动率 (模拟日内波动率)
    vol_daily = roller_std(rets, weriod, weriod, method)

    # 计算波动率的均值和标准差 (用于变异系数)
    vol_mean = roller_mean(vol_daily, weriod, weriod, method)
    vol_std = roller_std(vol_daily, weriod, weriod, method)

    # 变异系数 CV = std / mean (安全除法)
    cv = safe_div(vol_std, vol_mean)

    # 取负值: CV越大(波动不稳定)因子值越小
    # 这样做多时选择波动稳定的品种
    core1 = -cv

    # 最终用 window 做平滑
    alpha = roller_mean(core1, window, window, method)

    return alpha

"""
因子: dz003 - 四分位差信号强度因子
来源: 东证期货 - 国债期货量价因子挖掘 (2022-07-12)
复现日期: 2026-01-02

基于小时级分钟数据的四分位差特征，表现最优的因子之一
"""
from lumina.impulse.fixed import *
import numpy as np


def ki008(high, low,  window, weriod, ewm=False):
    """
    四分位差信号强度因子

    核心逻辑：
    1. 计算滚动窗口内的价格四分位差（IQR）
    2. 应用sigmoid归一化
    3. 取log10变换增强信号

    研报表达式: log10(sig(X15))
    其中 X15 = 小时内分钟价格的四分位差

    参数:
        close: 收盘价序列
        high: 最高价序列
        low: 最低价序列
        window: 外层平滑窗口
        weriod: 四分位差计算窗口（如60表示60分钟）
        ewm: 是否使用指数加权

    返回:
        因子值（负向信号，国债期货反转效应）

    来源研报: 东证期货量价因子挖掘
    表现: 夏普率1.88, 年化收益18.8% (T 窗口2 Alpha_6)
    """
    method = 'ewm' if ewm else 'rolling'

    # 使用高低价差作为IQR的代理指标（在分钟数据不可用时）
    price_range = high - low

    # 计算滚动IQR代理指标
    iqr_proxy = roller_mean(price_range, weriod, weriod, method)

    # 标准化
    iqr_std = (iqr_proxy - roller_mean(iqr_proxy, weriod*2, weriod*2, method)) / (roller_std(iqr_proxy, weriod*2, weriod*2, method) + 1e-8)
    iqr_sig = 1.0 / (1.0 + np.exp(-iqr_std))

    # log10变换
    iqr_signal = safe_log(iqr_sig + 1e-8, drift=10)
    iqr_signal = iqr_signal / np.log(10)

    # 最终平滑
    alpha = roller_mean(iqr_signal, window, 1, method)

    # 反转效应：取负值
    return -alpha

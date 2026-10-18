# -*- encoding:utf-8 -*-
"""
gf001: SRJV 上下行跳跃波动不对称因子 (Signed Realized Jump Volatility)

来源: 广发证券 - 高频数据因子研究系列九：基于股价跳跃模型的因子研究

原理:
基于跳跃-扩散模型，将已实现波动率分解为连续波动和跳跃波动，
再将跳跃波动按方向分解为上行跳跃和下行跳跃，计算其不对称性。

公式:
1. 已实现波动率: RV = Σ r²
2. 积分波动率估计 (三幂次变差): IV = μ^(-3) * Σ |r_i|^(2/3) * |r_{i-1}|^(2/3) * |r_{i-2}|^(2/3)
3. 上行已实现波动率: RV+ = Σ r² * I_{r>0}
4. 下行已实现波动率: RV- = Σ r² * I_{r<0}
5. 上行跳跃波动: RJVP = max(RV+ - IV/2, 0)
6. 下行跳跃波动: RJVN = max(RV- - IV/2, 0)
7. SRJV = RJVP - RJVN

因子解读:
- SRJV > 0: 正向跳跃多于负向跳跃，看多信号
- SRJV < 0: 负向跳跃多于正向跳跃，看空信号
- 原始研报中为负向因子(IC为负)，本实现取反使其为正向因子

应用:
- 高频数据因子，捕捉跳跃波动的方向性不对称
- 周度因子表现最佳(周度均值)
"""
import numpy as np
from scipy.special import gamma
from lumina.impulse.fixed import *


def ki012(close, window, weriod, ewm=False):
    """
    SRJV 上下行跳跃波动不对称因子

    参数:
        close: 收盘价 (1分钟数据)
        window: 外层平滑窗口
        weriod: 日内周期 (如240分钟/天)
        ewm: 是否使用指数加权

    返回:
        alpha: SRJV 因子值 (已取反，高值=看多)
    """
    method = 'ewm' if ewm else 'rolling'

    # 1. 计算对数收益率
    returns = safe_log(close)

    # 2. 计算收益率平方 (用于后续上行下行分离)
    returns_sq = returns ** 2

    # 3. 计算三幂次变差来估计积分波动率 IV
    # μ_{2/3} = 2^(1/3) * Γ(5/6) / Γ(1/2)
    m = 2.0 / 3.0
    mu_m = (2 ** (m / 2)) * gamma((m + 1) / 2) / gamma(0.5)

    # |r|^(2/3)
    abs_r_pow = np.abs(returns) ** m

    # 三幂次变差: |r_i|^m * |r_{i-1}|^m * |r_{i-2}|^m
    # 使用滚动乘积近似
    r_lag1 = abs_r_pow.shift(1)
    r_lag2 = abs_r_pow.shift(2)
    tripower = abs_r_pow * r_lag1 * r_lag2

    # IV estimate = μ^(-3) * Σ tripower
    # 注: μ^(-2/m) = μ^(-3) for m=2/3
    iv_raw = roller_sum(tripower, weriod, weriod, method)
    iv = (mu_m ** (-3)) * iv_raw

    # 4. 计算上行和下行已实现波动率
    # RV+ = Σ r² * I_{r>0}
    positive_mask = (returns > 0).astype(float)
    negative_mask = (returns < 0).astype(float)

    rv_up = roller_sum(returns_sq * positive_mask, weriod, weriod, method)
    rv_down = roller_sum(returns_sq * negative_mask, weriod, weriod, method)

    # 5. 计算上行和下行跳跃波动
    # RJVP = max(RV+ - IV/2, 0)
    # RJVN = max(RV- - IV/2, 0)
    rjvp = (rv_up - iv / 2).clip(lower=0)
    rjvn = (rv_down - iv / 2).clip(lower=0)

    # 6. SRJV = RJVP - RJVN
    srjv = rjvp - rjvn

    # 7. 取反 (原始因子IC为负，取反后高值=看多)
    core1 = -srjv

    # 8. 最终用 window 做平滑
    alpha = roller_mean(core1, window, window, method)

    return alpha

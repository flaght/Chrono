# -*- encoding:utf-8 -*-
"""
zs001: VPIN 指令流毒性因子 (Volume-synchronized Probability of Informed trading)

来源: 招商证券 - "琢璞"系列报告之十七：高频数据中的知情交易（二）

原理:
1. 将交易量按价格变化方向分类为买入量和卖出量
2. 计算指令不平衡程度 OI = |V_buy - V_sell|
3. VPIN = 指令不平衡占总交易量的比例

分类方法 (Bulk Volume Classification):
- V_buy = V * Z((P_t - P_{t-1}) / σ_ΔP)
- V_sell = V - V_buy
- Z 是标准正态分布的累积分布函数

应用:
- VPIN 高值表示市场存在大量知情交易，流动性风险增加
- 可用于预测波动率和极端行情
"""
import numpy as np
from scipy.stats import norm
from lumina.impulse.fixed import *


def ki032(close, volume, window, weriod, ewm=False):
    """
    VPIN 指令流毒性因子

    参数:
        close: 收盘价 (1分钟数据)
        volume: 成交量 (1分钟数据)
        window: 外层平滑窗口
        weriod: 计算周期 (用于估计价格变化标准差)
        ewm: 是否使用指数加权

    返回:
        alpha: VPIN 因子值 (高值表示高毒性/高知情交易概率)
    """
    method = 'ewm' if ewm else 'rolling'

    # 1. 计算价格变化
    price_change = close.diff()

    # 2. 计算价格变化的滚动标准差
    sigma_dp = roller_std(price_change, weriod, weriod, method)

    # 3. 标准化价格变化
    z_score = safe_div(price_change, sigma_dp)

    # 4. 使用标准正态CDF将交易量分类为买入/卖出
    # V_buy = V * Φ(z_score), V_sell = V * (1 - Φ(z_score))
    # 注意: norm.cdf 对 DataFrame 可直接应用
    buy_prob = z_score.apply(norm.cdf)

    v_buy = volume * buy_prob
    v_sell = volume * (1 - buy_prob)

    # 5. 计算指令不平衡
    order_imbalance = np.abs(v_buy - v_sell)

    # 6. 计算 VPIN = 滚动平均不平衡 / 滚动平均交易量
    avg_oi = roller_mean(order_imbalance, weriod, weriod, method)
    avg_vol = roller_mean(volume, weriod, weriod, method)

    vpin = safe_div(avg_oi, avg_vol)

    # 7. 最终用 window 做平滑
    alpha = roller_mean(vpin, window, window, method)

    return alpha

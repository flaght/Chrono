# -*- encoding:utf-8 -*-
"""
fz001: 勇攀高峰因子 (Climb Peak Factor)

来源: 方正证券 - 多因子选股系列研究之三：个股波动率的变动及"勇攀高峰"因子构建

原理:
1. 当波动异常高时，投资者风险厌恶快速增大
2. 能在高波动时提供充足风险补偿的资产，展现出非凡能力
3. 通过收益波动比(RVR)与波动率在高波动时段的协方差来衡量

构建步骤:
1. 计算"更优波动率" - 使用OHLC捕捉分钟内价格变动
2. 计算收益波动比 RVR = 收益率 / 更优波动率
3. 筛选高波动时段: 波动率 >= mean + std
4. 计算高波动时段内 RVR 与波动率的协方差
5. 取滚动均值和标准差，等权合成

因子解读:
- 正向因子，值越大越好
- 高值表示资产在高波动时能提供充足的风险补偿
"""
import numpy as np
from lumina.impulse.fixed import *


def ki011(high, low, open_, close, window, weriod, ewm=False):
    """
    勇攀高峰因子

    参数:
        high: 最高价 (1分钟数据)
        low: 最低价 (1分钟数据)
        open_: 开盘价 (1分钟数据)
        close: 收盘价 (1分钟数据)
        window: 外层平滑窗口
        weriod: 日内周期 (如240分钟/天)
        ewm: 是否使用指数加权

    返回:
        alpha: 勇攀高峰因子值 (正向因子，高值=看多)
    """
    method = 'ewm' if ewm else 'rolling'

    # 1. 计算收益率 (对数收益)
    returns = safe_log(close)
    # 2. 计算"更优波动率" - 使用Garman-Klass风格的波动率估计
    # σ² = 0.5 * (H-L)² - (2*ln(2)-1) * (C-O)²
    # 简化版: σ = sqrt(0.5*(H-L)² + 0.5*(C-O)²) / prev_close
    prev_close = close.shift(1)
    range_sq = (high - low) ** 2
    oc_sq = (close - open_) ** 2
    better_vol = np.sqrt(0.5 * range_sq + 0.5 * oc_sq)
    better_vol = safe_div(better_vol, prev_close)  # 标准化

    # 3. 计算收益波动比 RVR = return / volatility
    rvr = safe_div(returns, better_vol + 1e-10)

    # 4. 计算日内波动率均值和标准差，用于筛选高波动时段
    vol_mean = roller_mean(better_vol, weriod, weriod, method)
    vol_std = roller_std(better_vol, weriod, weriod, method)
    vol_threshold = vol_mean + vol_std

    # 5. 高波动时段掩码
    high_vol_mask = (better_vol >= vol_threshold).astype(float)

    # 6. 仅保留高波动时段的RVR和波动率
    rvr_high = rvr * high_vol_mask
    vol_high = better_vol * high_vol_mask

    # 7. 计算高波动时段的协方差代理
    # 协方差 ≈ E[XY] - E[X]*E[Y]
    # 由于仅关注高波动时段，用加权方式处理
    rvr_vol_product = rvr_high * vol_high

    # 滚动计算协方差的代理变量
    mean_product = roller_mean(rvr_vol_product, weriod, weriod, method)
    mean_rvr = roller_mean(rvr_high, weriod, weriod, method)
    mean_vol = roller_mean(vol_high, weriod, weriod, method)

    # 高波动时段计数 (用于标准化)
    count_high = roller_sum(high_vol_mask, weriod, 1, method)
    count_high = count_high.clip(lower=1)

    # 协方差 = E[XY] - E[X]E[Y]，调整计数
    cov_proxy = mean_product - safe_div(mean_rvr * mean_vol, count_high)

    # 8. 计算滚动均值和标准差 (月均攀登 + 月稳攀登)
    cov_mean = roller_mean(cov_proxy, weriod, weriod, method)  # 月均攀登
    cov_std = roller_std(cov_proxy, weriod, weriod, method)    # 月稳攀登

    # 9. 等权合成: (均值 + 标准差) / 2，需要标准化
    # 由于两者量纲可能不同，先各自标准化
    cov_mean_norm = safe_div(cov_mean, roller_std(cov_mean, weriod, weriod, method) + 1e-10)
    cov_std_norm = safe_div(cov_std, roller_std(cov_std, weriod, weriod, method) + 1e-10)

    core1 = (cov_mean_norm + cov_std_norm) / 2

    # 10. 最终用 window 做平滑
    alpha = roller_mean(core1, window, window, method)

    return alpha

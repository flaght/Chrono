import numpy as np
import pandas as pd
from lumina.impulse.fixed import *

def tmp003(close, openint, window, fast, slow, ewm=False,
           gamma=1.0, alpha=0.5, lambda_=0.05, persistence_span=10):
    """
    基于四象限状态强度编码的因子。
    参数：
        close: 收盘价宽表
        openint: 持仓量宽表
        window: 最终平滑窗口
        fast: 标准化滚动窗口（W）
        slow: 聚合窗口（L）
        ewm: 是否使用指数加权
        gamma, alpha, lambda_, persistence_span: 辅助参数（固定默认值）
    """
    method = 'ewm' if ewm else 'rolling'

    # Step 1: 分钟收益与持仓量变化
    r = safe_shift(close, 1)          # 对数收益
    oi = openint - openint.shift(1)   # 持仓量变化

    # Step 2: 滚动标准化
    r_mean = roller_mean(r, fast, 1, method)
    r_std = roller_std(r, fast, 1, method)
    r_std_safe = safe_div(r - r_mean, r_std)

    oi_mean = roller_mean(oi, fast, 1, method)
    oi_std = roller_std(oi, fast, 1, method)
    oi_std_safe = safe_div(oi - oi_mean, oi_std)

    # Step 3: 四象限状态分类
    up_oi  = (r_std_safe > 0) & (oi_std_safe > 0)   # 上涨增仓
    up_noi = (r_std_safe > 0) & (oi_std_safe < 0)   # 上涨减仓
    dn_oi  = (r_std_safe < 0) & (oi_std_safe > 0)   # 下跌增仓
    dn_noi = (r_std_safe < 0) & (oi_std_safe < 0)   # 下跌减仓

    # Step 4: 状态方向符号
    sign = pd.DataFrame(0, index=close.index, columns=close.columns)
    sign = sign.mask(up_oi, 1.0)
    sign = sign.mask(up_noi, -1.0)
    sign = sign.mask(dn_oi, -1.0)
    sign = sign.mask(dn_noi, 1.0)

    # Step 5: 强度权重（非线性压缩）
    w = np.tanh(gamma * oi_std_safe.abs())

    # Step 6: 状态持续性（用 ewm 平滑“是否连续”的布尔指示器近似）
    state = pd.DataFrame(0, index=close.index, columns=close.columns)
    state = state.mask(up_oi, 1)
    state = state.mask(up_noi, 2)
    state = state.mask(dn_oi, 3)
    state = state.mask(dn_noi, 4)
    same = (state == state.shift(1))          # 是否与前一bar状态相同
    p_t = roller_mean(same.astype(float), persistence_span, 1, 'ewm')

    # Step 7: 瞬时状态强度
    F_state = sign * w * (1 + alpha * p_t)

    # Step 8: 聚合（滚动平均近似加权求和）
    F = roller_mean(F_state, slow, 1, method)

    # 最终平滑（框架硬性要求）
    alpha_out = roller_mean(F, window, 1, method)
    return alpha_out
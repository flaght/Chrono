# -*- encoding:utf-8 -*-
"""
信息分布涨跌幅因子 URet' (西南证券 - 动量因子系列)
来源: "求索动量因子"系列研究（四）：反应不足or反应过度？从信息分布到动量反转

公式:
1. 计算每日成交量的标准差/均值作为信息分布代理 Z
2. RetPart1 = mean(Z值最小的days个交易日涨跌幅) - 信息分布最均匀
3. RetPart5 = mean(Z值最大的days个交易日涨跌幅) - 信息分布最不均匀
4. URet' = RetPart5 - RetPart1

简化实现: 使用成交量波动率作为信息分布代理
- 成交量波动大 → 信息分布不均匀 → 反转效应强
"""
from lumina.impulse.fixed import *


def ki030(close, volume, window, weriod, ewm=False):
    """
    信息分布涨跌幅因子

    参数:
        close: 收盘价序列
        volume: 成交量序列
        window: 外层平滑窗口
        weriod: 回望周期
        ewm: 是否使用指数加权

    返回:
        alpha: 因子值
    """
    method = 'ewm' if ewm else 'rolling'

    # 计算收益率
    rets = safe_log(close)

    # 计算成交量的变异系数 Z = std(volume) / mean(volume)
    # 作为信息分布均匀程度的代理
    vol_std = roller_std(volume, weriod, weriod, method)
    vol_mean = roller_mean(volume, weriod, weriod, method)
    info_dist = safe_div(vol_std, vol_mean)

    # 信息分布与收益率的交互
    # 信息分布越不均匀(Z越大), 收益反转效应越强
    core = rets * info_dist

    # 外层平滑
    alpha = roller_mean(core, window, window, method)

    return alpha

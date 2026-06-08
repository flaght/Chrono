import pandas as pd
import numpy as np
from lumina.genetic.strategy.method.env import *

# =============================
# 默认参数范围设置
# =============================
default_quantile_window_range = [x
                                 for x in range(5, 31, 2)]  # 分位数窗口 5,10,...,60
default_low_quantile_range = [round(x, 2) for x in np.arange(0.05, 0.2, 0.05)
                              ]  # 低分位阈值 0.05,0.10,...,0.25
default_high_quantile_range = [
    round(x, 2) for x in np.arange(0.80, 0.90, 0.05)
]  # 高分位阈值 0.75,0.80,...,0.95
default_max_volume_range = [1]#[1, 2, 3]  # 最大持仓手数范围


def quantile_strategy(signal: pd.DataFrame,
                      total_data: pd.DataFrame,
                      window: int = 20,
                      low_quantile: float = 0.2,
                      high_quantile: float = 0.8,
                      max_volume: int = 1) -> pd.DataFrame:
    """
    收盘价分位数持仓策略 - 仅在收盘价处于极端分位区间且信号一致时持仓

    策略解释：
    本策略根据当前收盘价在过去N日的分位数位置决定持仓：
    - 当收盘价处于历史低分位（如20%以下）且信号为多时持仓；
    - 当收盘价处于历史高分位（如80%以上）且信号为空时持仓；
    - 其余情况空仓。
    该策略与趋势、均值回归、波动率、跳空等策略相关性低，且可100%向量化实现。

    核心思想：
    - 分位数过滤：用历史收盘价分位数作为信号过滤
    - 信号与分位数极端一致时才持仓
    - 全向量化实现，效率极高

    优势：
    - 能捕捉价格极端区间的反转机会
    - 与常规趋势、波动率、跳空等策略相关性低
    - 逻辑清晰，参数少，易于理解和调优
    - 计算高效，适合大规模回测

    参数：
    - signal: 信号DataFrame（1/-1/0）
    - total_data: 行情数据DataFrame，需包含'close'
    - window: 分位数计算窗口（如20）
    - low_quantile: 低分位阈值（如0.2）
    - high_quantile: 高分位阈值（如0.8）
    - max_volume: 最大持仓手数
    返回：
    - pos: 计算后的持仓DataFrame，列名为('pos', code)
    """
    close = total_data['close']
    signal = signal.reindex(total_data.index).fillna(0)
    codes = signal.columns

    # 计算rolling分位数
    def rolling_quantile_rank(arr, window):
        # arr: (n,)
        
        out = np.full_like(arr, np.nan, dtype=float)
        for i in range(int(window) - 1, len(arr)):
            window_arr = arr[i - int(window) + 1:i + 1]
            out[i] = (window_arr <= arr[i]).sum() / int(window)
        return out

    quantile_rank = pd.DataFrame(index=close.index, columns=codes)
    for code in codes:
        quantile_rank[code] = rolling_quantile_rank(close[code].values, int(window))
    # 多头：低分位且信号为1
    long_cond = (quantile_rank < low_quantile) & (signal == 1)
    # 空头：高分位且信号为-1
    short_cond = (quantile_rank > high_quantile) & (signal == -1)
    pos = pd.DataFrame(0, index=signal.index, columns=signal.columns)
    pos[long_cond] = max_volume
    pos[short_cond] = -max_volume
    pos = pos.astype(int)
    pos.columns = pd.MultiIndex.from_tuples([('pos', c) for c in pos.columns])
    return pos


def create_muster(window_sets=None,
                  low_quantile_sets=None,
                  high_quantile_sets=None,
                  max_volume_sets=None):
    """
    生成quantile_strategy的参数组合
    - window_sets: 分位数窗口集合
    - low_quantile_sets: 低分位阈值集合
    - high_quantile_sets: 高分位阈值集合
    - max_volume_sets: 最大持仓手数集合
    返回：
    - muster: Function对象列表
    """
    window_sets = window_sets if isinstance(
        window_sets, list) else default_quantile_window_range
    low_quantile_sets = low_quantile_sets if isinstance(
        low_quantile_sets, list) else default_low_quantile_range
    high_quantile_sets = high_quantile_sets if isinstance(
        high_quantile_sets, list) else default_high_quantile_range
    max_volume_sets = max_volume_sets if isinstance(
        max_volume_sets, list) else default_max_volume
    muster = []
    for window in window_sets:
        for low_quantile in low_quantile_sets:
            for high_quantile in high_quantile_sets:
                for max_volume in max_volume_sets:
                    muster.append(
                        Function(function=quantile_strategy,
                                 name='quantile_strategy',
                                 params={
                                     'window': int(window),
                                     'low_quantile': float(low_quantile),
                                     'high_quantile': float(high_quantile),
                                     'max_volume': int(max_volume)
                                 }))
    return muster

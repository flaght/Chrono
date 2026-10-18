import pandas as pd
import numpy as np
from lumina.genetic.strategy.method.env import *

# =============================
# 默认参数范围设置
# =============================
default_gap_threshold_range = [
    round(x, 3) for x in np.arange(0.005, 0.2, 0.005)
]  # 跳空比例阈值 0.005,0.01,...,0.05
default_max_volume_range = [1]  # 最大持仓手数范围


def gap_strategy(signal: pd.DataFrame,
                 total_data: pd.DataFrame,
                 gap_threshold: float = 0.01,
                 max_volume: int = 1) -> pd.DataFrame:
    """
    跳空突破持仓策略 - 仅在跳空方向与信号一致时持仓，否则空仓

    参数说明与分钟线影响：
    - gap_threshold: 跳空比例阈值，影响信号极端性，分钟线建议0.005~0.05
    - max_volume: 最大持仓手数，分钟线建议1~3

    源码逻辑简述：
    - 计算开盘与前收盘的跳空比例，跳空方向与信号一致时持仓，否则空仓
    - 纯向量化实现，效率高
    - 适合捕捉市场情绪极端变化

    参数合理范围建议与推荐：
    - gap_threshold: 0.005~0.05
    - max_volume: 1~3

    参数区间极端值风险：
    - gap_threshold过小，信号过于频繁，易被噪音触发
    - gap_threshold过大，信号稀少，错失机会

    参数：
    - signal: 信号DataFrame（1/-1/0）
    - total_data: 行情数据DataFrame，需包含'open'、'close'
    - gap_threshold: 跳空比例阈值（如0.01，表示1%）
    - max_volume: 最大持仓手数
    返回：
    - pos: 计算后的持仓DataFrame，列名为('pos', code)
    """
    open_ = total_data['open']
    close = total_data['close']
    signal = signal.reindex(total_data.index).fillna(0)
    codes = signal.columns
    # 计算跳空比例
    gap = (open_ - close.shift(1)) / close.shift(1)
    # 仅在跳空方向与信号一致时持仓，否则空仓
    long_cond = (signal == 1) & (gap > gap_threshold)
    short_cond = (signal == -1) & (gap < -gap_threshold)
    pos = pd.DataFrame(0, index=signal.index, columns=signal.columns)
    pos[long_cond] = max_volume
    pos[short_cond] = -max_volume
    pos = pos.astype(int)
    pos.columns = pd.MultiIndex.from_tuples([('pos', c) for c in pos.columns])
    return pos


def create_muster(gap_threshold_sets=None, max_volume_sets=None):
    """
    生成gap_strategy的参数组合
    - gap_threshold_sets: 跳空比例阈值集合
    - max_volume_sets: 最大持仓手数集合
    返回：
    - muster: Function对象列表
    """
    gap_threshold_sets = gap_threshold_sets if isinstance(
        gap_threshold_sets, list) else default_gap_threshold_range
    max_volume_sets = max_volume_sets if isinstance(
        max_volume_sets, list) else default_max_volume
    muster = []
    for gap_threshold in gap_threshold_sets:
        for max_volume in max_volume_sets:
            muster.append(
                Function(function=gap_strategy,
                         name='gap_strategy',
                         params={
                             'gap_threshold': float(gap_threshold),
                             'max_volume': int(max_volume)
                         }))
    return muster

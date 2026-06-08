import pandas as pd
import numpy as np
from lumina.genetic.strategy.method.env import *

# =============================
# 默认参数范围设置
# =============================
default_rollingstddev_window_range = [x for x in range(5, 31, 5)
                                      ]  # 标准差窗口 5,10,...,60
default_std_threshold_range = [
    round(x, 3) for x in np.arange(0.005, 0.03, 0.005)
]  # 标准差阈值 0.005,0.01,...,0.05
default_max_volume_range = [1]#[1, 2, 3]  # 最大持仓手数范围


def rollingstddev_strategy(signal: pd.DataFrame,
                           total_data: pd.DataFrame,
                           window: int = 20,
                           std_threshold: float = 0.01,
                           max_volume: int = 1) -> pd.DataFrame:
    """
    滚动标准差策略 - 仅在波动率极端且信号一致时持仓

    策略解释：
    计算N窗口收盘价标准差，标准差高于阈值且信号为多时持仓（高波动做多）；低于阈值且信号为空时持仓（低波动做空）；其余空仓。
    该策略与趋势、均值回归、动量等策略相关性低，且可100%向量化实现。

    核心思想：
    - 波动率过滤：用N窗口标准差作为信号过滤
    - 信号与波动率极端一致时才持仓
    - 全向量化实现，效率极高

    优势：
    - 能捕捉高波动或低波动市场机会
    - 与常规趋势、均值回归、动量等策略相关性低
    - 逻辑清晰，参数少，易于理解和调优
    - 计算高效，适合大规模回测

    参数：
    - signal: 信号DataFrame（1/-1/0）
    - total_data: 行情数据DataFrame，需包含'close'
    - window: 标准差计算窗口（如20，单位：bar/分钟）
    - std_threshold: 标准差阈值
    - max_volume: 最大持仓手数
    返回：
    - pos: 计算后的持仓DataFrame，列名为('pos', code)
    """
    close = total_data['close']
    signal = signal.reindex(total_data.index).fillna(0)
    codes = signal.columns
    stddev = close.rolling(int(window)).std()
    # 多头：标准差高于阈值且信号为1
    long_cond = (stddev > std_threshold) & (signal == 1)
    # 空头：标准差低于阈值且信号为-1
    short_cond = (stddev < std_threshold) & (signal == -1)
    pos = pd.DataFrame(0, index=signal.index, columns=signal.columns)
    pos[long_cond] = max_volume
    pos[short_cond] = -max_volume
    pos = pos.astype(int)
    pos.columns = pd.MultiIndex.from_tuples([('pos', c) for c in pos.columns])
    return pos


def create_muster(window_sets=None,
                  std_threshold_sets=None,
                  max_volume_sets=None):
    """
    生成rollingstddev_strategy的参数组合
    - window_sets: 标准差窗口集合
    - std_threshold_sets: 标准差阈值集合
    - max_volume_sets: 最大持仓手数集合
    返回：
    - muster: Function对象列表
    """
    window_sets = window_sets if isinstance(
        window_sets, list) else default_rollingstddev_window_range
    std_threshold_sets = std_threshold_sets if isinstance(
        std_threshold_sets, list) else default_std_threshold_range
    max_volume_sets = max_volume_sets if isinstance(
        max_volume_sets, list) else default_max_volume
    muster = []
    for window in window_sets:
        for std_threshold in std_threshold_sets:
            for max_volume in max_volume_sets:
                muster.append(
                    Function(function=rollingstddev_strategy,
                             name='rollingstddev_strategy',
                             params={
                                 'window': int(window),
                                 'std_threshold': float(std_threshold),
                                 'max_volume': int(max_volume)
                             }))
    return muster

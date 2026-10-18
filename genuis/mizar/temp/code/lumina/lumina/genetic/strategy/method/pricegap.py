import pandas as pd
import numpy as np
from lumina.genetic.strategy.method.env import *

# =============================
# 默认参数范围设置
# =============================
default_pricegap_window_range = [x
                                 for x in range(5, 30, 2)]  # 跳变窗口 5,10,...,60
default_gap_threshold_range = [
    round(x, 3) for x in np.arange(0.005, 0.03, 0.005)
]  # 跳变阈值 0.005,0.01,...,0.05
default_max_volume_range = [1]#[1, 2, 3]  # 最大持仓手数范围


def pricegap_strategy(signal: pd.DataFrame,
                      total_data: pd.DataFrame,
                      window: int = 20,
                      gap_threshold: float = 0.01,
                      max_volume: int = 1) -> pd.DataFrame:
    """
    价格跳变策略 - 仅在价格大幅波动或极端平稳且信号一致时持仓

    策略解释：
    计算当前bar收盘价与N窗口前收盘价的绝对差值，差值超过阈值且信号为多时持仓（大幅波动做多）；低于阈值且信号为空时持仓（平稳做空）；其余空仓。
    该策略与趋势、均值回归、动量等策略相关性低，且可100%向量化实现。

    核心思想：
    - 跳变过滤：用N窗口价格绝对差值作为信号过滤
    - 信号与价格极端一致时才持仓
    - 全向量化实现，效率极高

    优势：
    - 能捕捉大幅波动或极端平稳市场机会
    - 与常规趋势、均值回归、动量等策略相关性低
    - 逻辑清晰，参数少，易于理解和调优
    - 计算高效，适合大规模回测

    参数：
    - signal: 信号DataFrame（1/-1/0）
    - total_data: 行情数据DataFrame，需包含'close'
    - window: 跳变计算窗口（如20，单位：bar/分钟）
    - gap_threshold: 跳变阈值
    - max_volume: 最大持仓手数
    返回：
    - pos: 计算后的持仓DataFrame，列名为('pos', code)
    """
    close = total_data['close']
    signal = signal.reindex(total_data.index).fillna(0)
    codes = signal.columns
    gap = (close - close.shift(int(window))).abs()
    # 多头：价格跳变大于阈值且信号为1
    long_cond = (gap > gap_threshold) & (signal == 1)
    # 空头：价格跳变小于阈值且信号为-1
    short_cond = (gap < gap_threshold) & (signal == -1)
    pos = pd.DataFrame(0, index=signal.index, columns=signal.columns)
    pos[long_cond] = max_volume
    pos[short_cond] = -max_volume
    pos = pos.astype(int)
    pos.columns = pd.MultiIndex.from_tuples([('pos', c) for c in pos.columns])
    return pos


def create_muster(window_sets=None,
                  gap_threshold_sets=None,
                  max_volume_sets=None):
    """
    生成pricegap_strategy的参数组合
    - window_sets: 跳变窗口集合
    - gap_threshold_sets: 跳变阈值集合
    - max_volume_sets: 最大持仓手数集合
    返回：
    - muster: Function对象列表
    """
    window_sets = window_sets if isinstance(
        window_sets, list) else default_pricegap_window_range
    gap_threshold_sets = gap_threshold_sets if isinstance(
        gap_threshold_sets, list) else default_gap_threshold_range
    max_volume_sets = max_volume_sets if isinstance(
        max_volume_sets, list) else default_max_volume
    muster = []
    for window in window_sets:
        for gap_threshold in gap_threshold_sets:
            for max_volume in max_volume_sets:
                muster.append(
                    Function(function=pricegap_strategy,
                             name='pricegap_strategy',
                             params={
                                 'window': int(window),
                                 'gap_threshold': float(gap_threshold),
                                 'max_volume': int(max_volume)
                             }))
    return muster

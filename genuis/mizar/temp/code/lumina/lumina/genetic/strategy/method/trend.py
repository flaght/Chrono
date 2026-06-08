import pandas as pd
import numpy as np
from lumina.genetic.strategy.method.env import *

# =============================
# 默认参数范围设置
# =============================
default_trend_window_range = [x for x in range(5, 30, 2)]  # 均线斜率窗口 5,10,...,60
default_max_volume_range = [1]#[1, 2, 3]  # 最大持仓手数范围


def trend_strategy(signal: pd.DataFrame,
                   total_data: pd.DataFrame,
                   window: int = 20,
                   max_volume: int = 1) -> pd.DataFrame:
    """
    趋势斜率策略 - 仅在均线斜率与信号一致时持仓

    策略解释：
    计算N窗口收盘价均线斜率，斜率为正且信号为多时持仓（顺势做多）；斜率为负且信号为空时持仓（顺势做空）；其余空仓。
    该策略与均值回归、动量、波动率等策略相关性低，且可100%向量化实现。

    核心思想：
    - 均线斜率：用N窗口均线斜率作为信号过滤
    - 信号与斜率方向一致时才持仓
    - 全向量化实现，效率极高

    优势：
    - 能捕捉趋势行情的顺势机会
    - 与常规均值回归、动量、波动率等策略相关性低
    - 逻辑清晰，参数少，易于理解和调优
    - 计算高效，适合大规模回测

    参数：
    - signal: 信号DataFrame（1/-1/0）
    - total_data: 行情数据DataFrame，需包含'close'
    - window: 均线斜率计算窗口（如20，单位：bar/分钟）
    - max_volume: 最大持仓手数
    返回：
    - pos: 计算后的持仓DataFrame，列名为('pos', code)
    """
    close = total_data['close']
    signal = signal.reindex(total_data.index).fillna(0)
    codes = signal.columns
    mean = close.rolling(int(window)).mean()
    slope = mean.diff(int(window))
    # 多头：斜率为正且信号为1
    long_cond = (slope > 0) & (signal == 1)
    # 空头：斜率为负且信号为-1
    short_cond = (slope < 0) & (signal == -1)
    pos = pd.DataFrame(0, index=signal.index, columns=signal.columns)
    pos[long_cond] = max_volume
    pos[short_cond] = -max_volume
    pos = pos.astype(int)
    pos.columns = pd.MultiIndex.from_tuples([('pos', c) for c in pos.columns])
    return pos


def create_muster(window_sets=None, max_volume_sets=None):
    """
    生成trend_strategy的参数组合
    - window_sets: 均线斜率窗口集合
    - max_volume_sets: 最大持仓手数集合
    返回：
    - muster: Function对象列表
    """
    window_sets = window_sets if isinstance(
        window_sets, list) else default_trend_window_range
    max_volume_sets = max_volume_sets if isinstance(
        max_volume_sets, list) else default_max_volume
    muster = []
    for window in window_sets:
        for max_volume in max_volume_sets:
            muster.append(
                Function(function=trend_strategy,
                         name='trend_strategy',
                         params={
                             'window': int(window),
                             'max_volume': int(max_volume)
                         }))
    return muster

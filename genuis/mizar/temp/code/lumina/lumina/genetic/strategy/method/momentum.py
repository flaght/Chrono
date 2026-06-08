import pandas as pd
import numpy as np
from lumina.genetic.strategy.method.env import *

# =============================
# 默认参数范围设置
# =============================
default_momentum_window_range = [x
                                 for x in range(5, 30, 2)]  # 动量窗口 5,10,...,60
default_max_volume_range = [1]#[1, 2, 3]  # 最大持仓手数范围


def momentum_strategy(signal: pd.DataFrame,
                      total_data: pd.DataFrame,
                      window: int = 10,
                      max_volume: int = 1) -> pd.DataFrame:
    """
    收盘价动量反转策略 - 仅在动量极端方向与信号一致时持仓

    策略解释：
    本策略根据N日收盘价动量（当前收盘价与N日前收盘价之差）进行反转操作：
    - 动量为负且信号为多时持仓（逢跌做多）；
    - 动量为正且信号为空时持仓（逢涨做空）；
    - 其余情况空仓。
    该策略与趋势、分位数、波动率等策略相关性低，且可100%向量化实现。

    核心思想：
    - 动量反转：用N日动量方向作为信号过滤
    - 信号与动量极端一致时才持仓
    - 全向量化实现，效率极高

    优势：
    - 能捕捉价格短期反转机会
    - 与常规趋势、分位数、波动率等策略相关性低
    - 逻辑清晰，参数少，易于理解和调优
    - 计算高效，适合大规模回测

    参数：
    - signal: 信号DataFrame（1/-1/0）
    - total_data: 行情数据DataFrame，需包含'close'
    - window: 动量计算窗口（如10）
    - max_volume: 最大持仓手数
    返回：
    - pos: 计算后的持仓DataFrame，列名为('pos', code)
    """
    close = total_data['close']
    signal = signal.reindex(total_data.index).fillna(0)
    codes = signal.columns
    # 计算N日动量
    momentum = close - close.shift(int(window))
    # 多头：动量为负且信号为1
    long_cond = (momentum < 0) & (signal == 1)
    # 空头：动量为正且信号为-1
    short_cond = (momentum > 0) & (signal == -1)
    pos = pd.DataFrame(0, index=signal.index, columns=signal.columns)
    pos[long_cond] = max_volume
    pos[short_cond] = -max_volume
    pos = pos.astype(int)
    pos.columns = pd.MultiIndex.from_tuples([('pos', c) for c in pos.columns])
    return pos


def create_muster(window_sets=None, max_volume_sets=None):
    """
    生成momentum_strategy的参数组合
    - window_sets: 动量窗口集合
    - max_volume_sets: 最大持仓手数集合
    返回：
    - muster: Function对象列表
    """
    window_sets = window_sets if isinstance(
        window_sets, list) else default_momentum_window_range
    max_volume_sets = max_volume_sets if isinstance(
        max_volume_sets, list) else default_max_volume
    muster = []
    for window in window_sets:
        for max_volume in max_volume_sets:
            muster.append(
                Function(function=momentum_strategy,
                         name='momentum_strategy',
                         params={
                             'window': int(window),
                             'max_volume': int(max_volume)
                         }))
    return muster

import pandas as pd
import numpy as np
from lumina.genetic.strategy.method.env import *

# =============================
# 默认参数范围设置
# =============================
default_turnover_window_range = [x for x in range(5, 31, 2)]  # 均值窗口 5,10,...,60
default_max_volume_range = [1]#[1, 2, 3]  # 最大持仓手数范围

def turnover_strategy(signal: pd.DataFrame,
                     total_data: pd.DataFrame,
                     window: int = 20,
                     max_volume: int = 1) -> pd.DataFrame:
    """
    成交量均值策略 - 仅在成交量偏离均值且信号一致时持仓

    策略解释：
    计算N窗口成交量均值，当前成交量高于均值且信号为多时持仓（放量做多）；低于均值且信号为空时持仓（缩量做空）；其余空仓。
    该策略与趋势、均值回归、动量等策略相关性低，且可100%向量化实现。

    核心思想：
    - 成交量均值：用N窗口成交量均值作为信号过滤
    - 信号与成交量偏离方向一致时才持仓
    - 全向量化实现，效率极高

    优势：
    - 能捕捉量价联动的市场机会
    - 与常规趋势、均值回归、动量等策略相关性低
    - 逻辑清晰，参数少，易于理解和调优
    - 计算高效，适合大规模回测

    参数：
    - signal: 信号DataFrame（1/-1/0）
    - total_data: 行情数据DataFrame，需包含'volume'
    - window: 均值计算窗口（如20，单位：bar/分钟）
    - max_volume: 最大持仓手数
    返回：
    - pos: 计算后的持仓DataFrame，列名为('pos', code)
    """
    volume = total_data['volume']
    signal = signal.reindex(total_data.index).fillna(0)
    codes = signal.columns
    mean = volume.rolling(int(window)).mean()
    # 多头：成交量高于均值且信号为1
    long_cond = (volume > mean) & (signal == 1)
    # 空头：成交量低于均值且信号为-1
    short_cond = (volume < mean) & (signal == -1)
    pos = pd.DataFrame(0, index=signal.index, columns=signal.columns)
    pos[long_cond] = max_volume
    pos[short_cond] = -max_volume
    pos = pos.astype(int)
    pos.columns = pd.MultiIndex.from_tuples([('pos', c) for c in pos.columns])
    return pos

def create_muster(window_sets=None, max_volume_sets=None):
    """
    生成turnover_strategy的参数组合
    - window_sets: 均值窗口集合
    - max_volume_sets: 最大持仓手数集合
    返回：
    - muster: Function对象列表
    """
    window_sets = window_sets if isinstance(window_sets, list) else default_turnover_window_range
    max_volume_sets = max_volume_sets if isinstance(max_volume_sets, list) else default_max_volume_range
    muster = []
    for window in window_sets:
        for max_volume in max_volume_sets:
            muster.append(
                Function(function=turnover_strategy,
                         name='turnover_strategy',
                         params={
                             'window': int(window),
                             'max_volume': int(max_volume)
                         }))
    return muster 
import pdb
import pandas as pd
import numpy as np
from lumina.genetic.strategy.method.env import *

# =============================
# 默认参数范围设置
# =============================
default_holding_period_range = [x for x in range(5, 31, 2)]  # 固定持有周期 5,10,...,30
default_max_volume_range = [1]  # 最大持仓手数范围


def holding_strategy(signal: pd.DataFrame, total_data: pd.DataFrame,
                     holding_period: int, max_volume: int) -> pd.DataFrame:
    """
    固定持仓周期策略 - 采用固定持有周期的基础持仓方法

    参数说明与分钟线影响：
    - holding_period: 固定持有周期，影响信号持仓时长，分钟线建议5~30
    - max_volume: 最大持仓手数，分钟线建议1~3

    源码逻辑简述：
    - T-1时刻信号为开仓依据，持有固定周期后平仓
    - T时刻持仓等于过去周期窗口内信号之和
    - 持仓数量受最大持仓手数限制
    - 纯向量化实现，效率高

    参数合理范围建议与推荐：
    - holding_period: 5~30
    - max_volume: 1~3

    参数区间极端值风险：
    - holding_period过小，信号过于频繁，易被噪音触发
    - holding_period过大，信号滞后，错失机会

    参数：
    - signal: 信号DataFrame
    - total_data: 总行情数据DataFrame
    - holding_period: 固定持有周期
    - max_volume: 最大持仓手数
    返回：
    - pos: 计算后的持仓DataFrame，列名为('pos', code)
    """
    holding_period = int(holding_period)
    max_volume = int(max_volume)
    # 平移获得T-1时刻的信号
    last_signal: pd.DataFrame = signal.reindex(
        total_data.index).shift(1).fillna(method='ffill').fillna(0)
    # T时刻持仓，等于过去持有周期窗口内的信号之和
    pos: pd.DataFrame = last_signal.rolling(int(holding_period)).sum().fillna(0)
    # 限制最大持仓数量
    pos = pos.clip(lower=-max_volume, upper=max_volume)
    columns = pd.MultiIndex.from_tuples([('pos', col) for col in pos.columns])
    pos.columns = columns
    return pos


def create_muster(holding_period_sets=None, max_volume_sets=None):
    """
    生成holding_strategy的参数组合
    - holding_period_sets: 固定持有周期集合
    - max_volume_sets: 最大持仓手数集合
    返回：
    - muster: Function对象列表
    """
    holding_period_sets = holding_period_sets if isinstance(
        holding_period_sets, list) else default_holding_period_range
    max_volume_sets = max_volume_sets if isinstance(
        max_volume_sets, list) else default_max_volume_range
    muster = []
    for holding_period in holding_period_sets:
        for max_volume in max_volume_sets:
            muster.append(
                Function(function=holding_strategy,
                        name='holding_strategy',
                         params={
                             'holding_period': int(holding_period),
                             'max_volume': int(max_volume)
                         }))
    return muster

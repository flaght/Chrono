import pandas as pd
from lumina.genetic.strategy.method.env import *

# =============================
# 默认参数范围设置
# =============================
default_max_volume_range = [1]#[1, 2, 3]  # 最大持仓手数范围


def simple_strategy(signal: pd.DataFrame, total_data: pd.DataFrame,
                    max_volume: int) -> pd.DataFrame:
    """
    简单持仓策略 - 采用上一时刻信号的基础持仓方法

    策略解释：
    该策略以T-1时刻的信号作为T时刻的持仓依据，逻辑极为直接，适合作为基线或基础持仓策略。

    核心思想：
    - 信号平移：T时刻持仓等于T-1时刻信号
    - 直接映射：信号直接决定持仓，无复杂处理
    - 便于对比：常用于与其他复杂策略做基准对比

    优势：
    - 极简明了：实现和理解都非常简单
    - 计算高效：无多余计算，速度快
    - 适用广泛：可作为回测基线或策略拼接的基础

    参数：
    - signal: 信号DataFrame，通常为多因子信号
    - total_data: 总行情数据DataFrame
    - max_volume: 最大持仓手数
    返回：
    - pos: 计算后的持仓DataFrame，列名为('pos', code)
    """
    max_volume = int(max_volume)
    # 平移获得T-1时刻的信号
    last_signal: pd.DataFrame = signal.reindex(
        total_data.index).shift(1).fillna(method='ffill').fillna(0)

    # T时刻持仓，等于T-1时刻的信号
    pos: pd.DataFrame = last_signal

    # 乘以固定交易手数后，保存到结果中
    pos = pos * max_volume

    columns = pd.MultiIndex.from_tuples([('pos', col) for col in pos.columns])
    pos.columns = columns

    return pos


def create_muster(max_volume_sets=None):
    """
    生成simple_strategy的参数组合
    - max_volume_sets: 最大持仓手数集合
    返回：
    - muster: Function对象列表
    """
    max_volume_sets = max_volume_sets if isinstance(
        max_volume_sets, list) else default_max_volume_range
    muster = []
    for max_volume in max_volume_sets:
        muster.append(
            Function(function=simple_strategy,
                    name='simple_strategy',
                     params={'max_volume': int(max_volume)}))
    return muster

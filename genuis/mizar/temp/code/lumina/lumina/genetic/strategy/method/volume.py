import pandas as pd
import numpy as np
from lumina.genetic.strategy.method.env import *

# =============================
# 默认参数范围设置
# =============================
default_volume_roll_num_range = [x for x in range(5, 31, 2)
                                 ]  # 成交量滚动窗口 5,10,...,60
default_max_volume_range = [1]#[1, 2, 3]  # 最大持仓手数范围


def volume_weighted_strategy(signal: pd.DataFrame, total_data: pd.DataFrame,
                             roll_num: int, max_volume: int) -> pd.DataFrame:
    """
    成交量加权持仓策略 - 根据成交量分布调整持仓

    策略解释：
    该策略通过成交量的滚动均值与当前成交量的比值，动态调整持仓，适合量价联动类策略。

    核心思想：
    - 成交量加权：持仓随成交量变化动态调整
    - 信号平移：T时刻持仓等于T-1时刻信号
    - 风险自适应：高成交量时持仓更激进，低成交量时更保守

    优势：
    - 量价联动：能捕捉市场活跃度变化
    - 适应性强：自动适应不同市场环境
    - 计算高效：实现简单，速度快

    参数：
    - signal: 信号DataFrame
    - total_data: 总行情数据DataFrame
    - roll_num: 成交量滚动窗口
    - max_volume: 最大持仓手数
    返回：
    - pos: 计算后的持仓DataFrame，列名为('pos', code)
    """
    roll_num = int(roll_num)
    max_volume = int(max_volume)
    last_signal = signal.reindex(total_data.index).shift(1).fillna(0)
    vol_ratio = total_data['volume'] / total_data['volume'].rolling(
        roll_num).mean()
    pos = last_signal.mul(vol_ratio, axis=0).clip(-max_volume, max_volume)
    pos = pos.fillna(0).round().astype(int)
    pos.columns = pd.MultiIndex.from_tuples([('pos', c) for c in pos.columns])
    return pos


def create_muster(roll_num_sets=None, max_volume_sets=None):
    """
    生成volume_weighted_strategy的参数组合
    - roll_num_sets: 成交量滚动窗口集合
    - max_volume_sets: 最大持仓手数集合
    返回：
    - muster: Function对象列表
    """
    roll_num_sets = roll_num_sets if isinstance(
        roll_num_sets, list) else default_volume_roll_num_range
    max_volume_sets = max_volume_sets if isinstance(
        max_volume_sets, list) else default_max_volume_range
    muster = []
    for roll_num in roll_num_sets:
        for max_volume in max_volume_sets:
            muster.append(
                Function(function=volume_weighted_strategy,
                         name='volume_weighted_strategy',
                         params={
                             'roll_num': int(roll_num),
                             'max_volume': int(max_volume)
                         }))
    return muster

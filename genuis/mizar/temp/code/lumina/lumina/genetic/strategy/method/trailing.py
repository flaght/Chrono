import pandas as pd
import numpy as np
import numba as nb
from lumina.genetic.strategy.method.env import *

# =============================
# 默认参数范围设置
# =============================
default_trailing_percent_range = [round(x, 3) for x in np.arange(0.005, 0.03, 0.005)]  # 止损百分比 0.005,0.01,...,0.05
default_max_volume_range = [1]#[1, 2, 3]  # 最大持仓手数范围


def trailing_strategy(signal: pd.DataFrame, total_data: pd.DataFrame,
                      trailing_percent: float,
                      max_volume: int) -> pd.DataFrame:
    """
    移动止损策略 - 采用百分比止损线的基础持仓方法

    策略解释：
    该策略以T-1时刻信号为开仓依据，采用移动止损（百分比）方式动态调整持仓，适合趋势跟踪类策略。

    核心思想：
    - 百分比止损：止损线随价格动态调整，防止大幅回撤
    - 信号平移：T时刻持仓等于T-1时刻信号
    - 动态跟踪：持仓随价格波动动态调整止损点

    优势：
    - 趋势跟踪：有助于捕捉大行情，减少小波动干扰
    - 计算高效：实现简单，速度快
    - 自动化强：参数化后可批量生成多组策略

    参数：
    - signal: 信号DataFrame
    - total_data: 总行情数据DataFrame
    - trailing_percent: 止损百分比
    - max_volume: 最大持仓手数
    返回：
    - pos: 计算后的持仓DataFrame，列名为('pos', code)
    """
    trailing_percent = float(trailing_percent)
    max_volume = int(max_volume)
    signal = signal.reindex(
        total_data.index).shift(1).fillna(method='ffill').fillna(0)
    codes = signal.columns.tolist()
    columns = pd.MultiIndex.from_tuples([('signal', col)
                                         for col in signal.columns])
    signal.columns = columns
    df = pd.concat([total_data[['high', 'low']], signal], join='outer', axis=1)
    cols = ['high', 'low', 'signal']
    pos_list = []
    for code in codes:
        pos_list.append(
            pd.Series(trailing_percent_code(arr=df.stack().xs(
                code, level=1)[cols].values,
                                            trailing_percent=trailing_percent,
                                            max_volume=max_volume),
                      name=code))
    pos = pd.concat(pos_list, axis=1).sort_index()
    columns = pd.MultiIndex.from_tuples([('pos', col) for col in pos.columns])
    pos.columns = columns
    pos.index = df.index
    return pos


@nb.njit(nogil=True, cache=False, fastmath=True)
def trailing_percent_code(arr: np.ndarray, trailing_percent: float,
                          max_volume: int) -> None:
    """
    移动止损持仓计算核心函数（Numba加速）

    策略解释：
    采用移动止损的交易策略，基于百分比动态调整止损线。
    T-1时刻信号出现后开仓，采用移动止损出场。

    核心思想：
    - 百分比止损：止损线随价格动态调整，防止大幅回撤
    - 动态跟踪：持仓随价格波动动态调整止损点
    - 信号直接映射：信号直接决定持仓

    优势：
    - 趋势跟踪：有助于捕捉大行情，减少小波动干扰
    - 计算高效：实现简单，速度快
    - 自动化强：参数化后可批量生成多组策略

    参数：
    - arr: 输入行情与信号数据，二维数组，列顺序为[high, low, signal]
    - trailing_percent: 止损百分比
    - max_volume: 最大持仓手数
    返回：
    - pos_data: 逐时刻持仓数组
    """
    high = arr[:, 0]
    low = arr[:, 1]
    signal = arr[:, 2]
    pos: int = 0
    pos_data = np.zeros_like(signal)
    trailing_high: float = 0  # 移动止损高点
    trailing_low: float = 0   # 移动止损低点
    # 遍历逐行执行持仓与止损逻辑
    for i in range(len(signal)):
        # 多头持仓
        if pos > 0:
            if signal[i] != signal[i] or signal[i] < 0:
                pos = 0
            if low[i] <= trailing_high * (1 - trailing_percent):
                pos = 0
        # 空头持仓
        elif pos < 0:
            if signal[i] != signal[i] or signal[i] > 0:
                pos = 0
            if high[i] >= trailing_low * (1 + trailing_percent):
                pos = 0
        # 无持仓
        else:
            if signal[i] > 0:
                pos = max_volume
            elif signal[i] < 0:
                pos = -max_volume
        # 计算移动止损价格
        if not pos:
            trailing_high = high[i]
            trailing_low = low[i]
        elif pos > 0:
            trailing_high = max(trailing_high, high[i])
            trailing_low = low[i]
        else:
            trailing_high = trailing_high
            trailing_low = min(trailing_low, low[i])
        pos_data[i] = pos
    return pos_data


def create_muster(trailing_percent_sets=None, max_volume_sets=None):
    """
    生成trailing_strategy的参数组合
    - trailing_percent_sets: 止损百分比集合
    - max_volume_sets: 最大持仓手数集合
    返回：
    - muster: Function对象列表
    """
    trailing_percent_sets = trailing_percent_sets if isinstance(
        trailing_percent_sets, list) else default_trailing_percent_range
    max_volume_sets = max_volume_sets if isinstance(
        max_volume_sets, list) else default_max_volume_range
    muster = []
    for trailing_percent in trailing_percent_sets:
        for max_volume in max_volume_sets:
            muster.append(
                Function(function=trailing_strategy,
                         name='trailing_strategy',
                         params={
                             'trailing_percent': float(trailing_percent),
                             'max_volume': int(max_volume)
                         }))
    return muster

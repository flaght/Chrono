import pandas as pd
import numpy as np
import numba as nb
from lumina.genetic.strategy.method.env import *

# =============================
# 默认参数范围设置
# =============================
default_atr_period_range = [x for x in range(5, 31, 5)]  # ATR周期范围 5,10,...,30
default_atr_multiplier_range = [x for x in range(2, 6)]  # ATR乘数范围 2,3,...,8
default_maN_range = [30, 60, 90, 120]  # 均线周期常用值
# 最大持仓手数范围（如需扩展可修改）
default_max_volume_range = [1]#[1, 2, 3]


# =============================
# 移动止损ATR策略核心函数（Numba加速）
# =============================
@nb.njit(nogil=True, cache=False, fastmath=True)
def trailing_percent_code_atr(arr: np.ndarray, atr_period: int,
                              atr_multiplier: float, max_volume: int,
                              maN: int) -> None:
    """
    移动止损ATR策略 - 逐行计算持仓信号（Numba加速，无法向量化）

    参数说明与分钟线影响：
    - atr_period: ATR计算周期，影响止损灵敏度，分钟线下建议10-30
    - atr_multiplier: ATR乘数，影响止损带宽，分钟线下建议2-8
    - max_volume: 最大持仓手数，分钟线下建议1-3
    - maN: 均线周期，影响趋势过滤，分钟线下建议30-120

    源码逻辑简述：
    - 逐行计算ATR和均线，根据信号和止损条件动态调整持仓
    - 多头：低于移动止损线或信号反转平仓
    - 空头：高于移动止损线或信号反转平仓
    - 均线过滤仅做辅助

    参数合理范围建议与推荐：
    - atr_period: 5~30
    - atr_multiplier: 2~8
    - max_volume: 1~3
    - maN: 30~120

    参数区间极端值风险：
    - atr_period过小，止损过于频繁，易被噪音触发
    - atr_period过大，止损滞后，丧失保护作用
    - atr_multiplier过小，易频繁止损
    - atr_multiplier过大，止损过宽，风险加大
    - maN过小，趋势过滤失效，过大则信号滞后

    实际调参建议：
    - 先固定max_volume=1，主调atr_period和atr_multiplier
    - maN建议与主流均线周期一致（如60、120）
    - 结合回测结果微调，关注极端行情下止损表现
    """
    high = arr[:, 0]
    low = arr[:, 1]
    close = arr[:, 2]
    signal = arr[:, 3]
    pos: int = 0
    pos_data = np.zeros_like(signal)
    trailing_high: float = 0  # 移动止损高点
    trailing_low: float = 0  # 移动止损低点
    atr: float = 0
    atr = np.zeros_like(signal)
    # 计算True Range
    true_range = np.maximum(high - low, np.abs(high - np.roll(close, 1)),
                            np.abs(low - np.roll(close, 1)))
    atr[0] = np.mean(true_range[:atr_period])
    for i in range(1, len(close)):
        if i < atr_period:
            atr[i] = np.mean(true_range[:i + 1])
        else:
            atr[i] = np.mean(true_range[i - atr_period + 1:i + 1])
    price_cost = 0.000
    # 逐行执行移动止损逻辑
    for i in range(len(signal)):
        # 均线过滤
        if i == 0:
            ma_val = np.mean(close[:i + 1])
        else:
            if i < maN + 1:
                ma_val = np.mean(close[:i])
            else:
                ma_val = np.mean(close[i - maN:i])
        # 多头持仓逻辑
        if pos > 0:
            if signal[i] != signal[i] or signal[i] < 0:
                pos = 0
            if low[i] <= trailing_high - atr[i] * atr_multiplier:
                pos = 0
            if close[i] < ma_val:
                pass
        # 空头持仓逻辑
        elif pos < 0:
            if signal[i] != signal[i] or signal[i] > 0:
                pos = 0
            if high[i] >= trailing_low + atr[i] * atr_multiplier:
                pos = 0
            if close[i] > ma_val:
                pass
        # 无持仓，根据信号开仓
        else:
            if signal[i] > 0:
                pos = max_volume
                price_cost = close[i]
            elif signal[i] < 0:
                pos = -max_volume
                price_cost = close[i]
        # 计算移动止损价格
        if not pos:
            trailing_high = high[i]
            trailing_low = low[i]
        elif pos > 0:
            trailing_high = max(trailing_high, high[i])
            trailing_low = low[i]
        elif pos < 0:
            trailing_high = trailing_high
            trailing_low = min(trailing_low, low[i])
        else:
            trailing_high = high[i]
            trailing_low = low[i]
        posnew = pos  # 必须把pos 保存起来当最新的持仓来交易
        pos_data[i] = pos
    return pos_data


# =============================
# 移动止损ATR策略 - pandas接口
# =============================
def trailing_atr_strategy(signal: pd.DataFrame,
                          total_data: pd.DataFrame,
                          atr_period: int,
                          atr_multiplier: float,
                          max_volume: int,
                          maN: int = 60) -> pd.DataFrame:
    """
    移动止损ATR策略 - pandas接口

    参数说明与分钟线影响：
    - atr_period: ATR计算周期，影响止损灵敏度，分钟线下建议10-30
    - atr_multiplier: ATR乘数，影响止损带宽，分钟线下建议2-8
    - max_volume: 最大持仓手数，分钟线下建议1-3
    - maN: 均线周期，影响趋势过滤，分钟线下建议30-120

    源码逻辑简述：
    - 逐行计算ATR和均线，根据信号和止损条件动态调整持仓
    - 多头：低于移动止损线或信号反转平仓
    - 空头：高于移动止损线或信号反转平仓
    - 均线过滤仅做辅助

    参数合理范围建议与推荐：
    - atr_period: 5~30
    - atr_multiplier: 2~8
    - max_volume: 1~3
    - maN: 30~120

    参数区间极端值风险：
    - atr_period过小，止损过于频繁，易被噪音触发
    - atr_period过大，止损滞后，丧失保护作用
    - atr_multiplier过小，易频繁止损
    - atr_multiplier过大，止损过宽，风险加大
    - maN过小，趋势过滤失效，过大则信号滞后

    实际调参建议：
    - 先固定max_volume=1，主调atr_period和atr_multiplier
    - maN建议与主流均线周期一致（如60、120）
    - 结合回测结果微调，关注极端行情下止损表现

    参数：
    - signal: 信号DataFrame（1/-1/0）
    - total_data: 行情数据DataFrame，需包含'high','low','close'
    - atr_period: ATR计算周期
    - atr_multiplier: ATR乘数
    - max_volume: 最大持仓手数
    - maN: 均线周期
    返回：
    - pos: 计算后的持仓DataFrame，列名为('pos', code)
    """
    atr_period = int(atr_period)
    max_volume = int(max_volume)
    atr_multiplier = float(atr_multiplier)
    maN = int(maN)
    signal = signal.reindex(
        total_data.index).shift(1).fillna(method='ffill').fillna(0)
    codes = signal.columns.tolist()
    columns = pd.MultiIndex.from_tuples([('signal', col)
                                         for col in signal.columns])
    signal.columns = columns
    df = pd.concat([total_data[['high', 'low', 'close']], signal],
                   join='outer',
                   axis=1)
    cols = ['high', 'low', 'close', 'signal']
    pos_list = []
    for code in codes:
        pos_list.append(
            pd.Series(trailing_percent_code_atr(
                arr=df.stack().xs(code, level=1)[cols].values,
                atr_period=int(atr_period),
                atr_multiplier=float(atr_multiplier),
                max_volume=int(max_volume),
                maN=int(maN)),
                      name=code))
    pos = pd.concat(pos_list, axis=1).sort_index()
    columns = pd.MultiIndex.from_tuples([('pos', col) for col in pos.columns])
    pos.columns = columns
    pos.index = df.index
    return pos


# =============================
# 参数组合生成器
# =============================
def create_muster(atr_period_sets=None,
                  atr_multiplier_sets=None,
                  maN_sets=None):
    """
    生成trailing_atr_strategy的参数组合
    - atr_period_sets: ATR周期集合
    - atr_multiplier_sets: ATR乘数集合
    - maN_sets: 均线周期集合
    返回：
    - muster: Function对象列表
    """
    atr_multiplier_sets = atr_multiplier_sets if isinstance(
        atr_multiplier_sets, list) else default_atr_multiplier_range
    atr_period_sets = atr_period_sets if isinstance(
        atr_period_sets, list) else default_atr_period_range
    maN_sets = maN_sets if isinstance(maN_sets, list) else default_maN_range
    muster = []
    for atr_period in atr_period_sets:
        for atr_multiplier in atr_multiplier_sets:
            for maN in maN_sets:
                for max_volume in default_max_volume:
                    muster.append(
                        Function(function=trailing_atr_strategy,
                                 name='trailing_atr_strategy',
                                 params={
                                     'atr_period': int(atr_period),
                                     'atr_multiplier': float(atr_multiplier),
                                     'max_volume': int(max_volume),
                                     'maN': int(maN)
                                 }))
    return muster

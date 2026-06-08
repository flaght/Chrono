import pandas as pd
import numpy as np
from lumina.genetic.strategy.method.env import *

# =============================
# 默认参数范围设置
# =============================
default_window_range = [x for x in range(10, 30, 5)]  # 最大回撤窗口 10,20,...,60
default_dd_threshold_range = [
    round(x, 3) for x in np.arange(0.01, 0.03, 0.01)
]  # 回撤阈值 0.01,0.02,...,0.05
default_max_volume_range = [1]#[1, 2, 3]  # 最大持仓手数范围


def drawdown_strategy(signal: pd.DataFrame,
                      total_data: pd.DataFrame,
                      window: int = 20,
                      dd_threshold: float = 0.01,
                      max_volume: int = 1) -> pd.DataFrame:
    """
    最大回撤策略 - 仅在极端回撤或极端平稳且信号一致时持仓

    参数说明与分钟线影响：
    - window: 最大回撤计算窗口，影响回撤灵敏度，分钟线建议10~60
    - dd_threshold: 最大回撤阈值，影响信号极端性，分钟线建议0.01~0.05
    - max_volume: 最大持仓手数，分钟线建议1~3

    源码逻辑简述：
    - 计算N窗口内最大回撤，回撤超过阈值且信号为多时持仓，回撤极小且信号为空时持仓，其余空仓
    - 纯向量化实现，效率高
    - 适合捕捉极端行情反弹或极端平稳做空

    参数合理范围建议与推荐：
    - window: 10~60
    - dd_threshold: 0.01~0.05
    - max_volume: 1~3

    参数区间极端值风险：
    - window过小，回撤计算不稳定，易受噪音影响
    - window过大，信号滞后，错失极端机会
    - dd_threshold过小，信号过于频繁，易被噪音触发
    - dd_threshold过大，信号稀少，错失机会

    参数：
    - signal: 信号DataFrame（1/-1/0）
    - total_data: 行情数据DataFrame，需包含'close'
    - window: 最大回撤计算窗口
    - dd_threshold: 最大回撤阈值
    - max_volume: 最大持仓手数
    返回：
    - pos: 计算后的持仓DataFrame，列名为('pos', code)
    """
    close = total_data['close']
    signal = signal.reindex(total_data.index).fillna(0)
    codes = signal.columns

    def rolling_max_drawdown(arr, window):
        out = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            window_arr = arr[i - window + 1:i + 1]
            maxp = np.maximum.accumulate(window_arr)
            dd = (window_arr - maxp) / maxp
            out[i] = dd.min()
        return out

    dd = pd.DataFrame(index=close.index, columns=codes)
    for code in codes:
        dd[code] = rolling_max_drawdown(close[code].values, int(window))
    # 多头：极端回撤且信号为1
    long_cond = (dd < -dd_threshold) & (signal == 1)
    # 空头：极端平稳且信号为-1
    short_cond = (dd > -dd_threshold) & (signal == -1)
    pos = pd.DataFrame(0, index=signal.index, columns=signal.columns)
    pos[long_cond] = max_volume
    pos[short_cond] = -max_volume
    pos = pos.astype(int)
    pos.columns = pd.MultiIndex.from_tuples([('pos', c) for c in pos.columns])
    return pos


def create_muster(window_sets=None,
                  dd_threshold_sets=None,
                  max_volume_sets=None):
    """
    生成drawdown_strategy的参数组合
    - window_sets: 最大回撤窗口集合
    - dd_threshold_sets: 最大回撤阈值集合
    - max_volume_sets: 最大持仓手数集合
    返回：
    - muster: Function对象列表
    """
    window_sets = window_sets if isinstance(window_sets,
                                            list) else default_window_range
    dd_threshold_sets = dd_threshold_sets if isinstance(
        dd_threshold_sets, list) else default_dd_threshold_range
    max_volume_sets = max_volume_sets if isinstance(
        max_volume_sets, list) else default_max_volume
    muster = []
    for window in window_sets:
        for dd_threshold in dd_threshold_sets:
            for max_volume in max_volume_sets:
                muster.append(
                    Function(function=drawdown_strategy,
                             name='drawdown_strategy',
                             params={
                                 'window': int(window),
                                 'dd_threshold': float(dd_threshold),
                                 'max_volume': int(max_volume)
                             }))
    return muster

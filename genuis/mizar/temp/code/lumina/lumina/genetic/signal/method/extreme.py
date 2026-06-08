import numpy as np
import pandas as pd
from lumina.genetic.rolling import rolling_zscore
from numba import njit
from lumina.genetic.signal.method.env import Function

@njit
def fast_extreme_count_window(arr, threshold):
    n = arr.shape[0]
    out = np.empty(n)
    for i in range(n):
        x = arr[i]
        high = np.sum(x > threshold)
        low = np.sum(x < -threshold)
        if high > low:
            out[i] = -1
        elif low > high:
            out[i] = 1
        else:
            out[i] = 0
    return out

default_rolling_range = [x for x in range(20, 60, 5)]
default_threshold_range = [round(x, 2) for x in list(np.arange(1.0, 2.5, 0.1))]


def extreme_count_signal(factor_data: pd.DataFrame,
                        roll_num: int = 20,
                        threshold: float = 0.0) -> pd.Series:
    """
    极端值计数信号模型 - 统计窗口内极端值出现频率，反映市场极端情绪
    
    策略解释：
    基于极端值理论，统计滚动窗口内标准化因子值超过正/负阈值的频率，反映市场极端情绪。当极端高频率时，认为市场过热，产生做空信号；极端低频率时，认为市场过冷，产生做多信号。
    
    核心思想：
    - 极端值计数：统计窗口内极端高/低值出现次数
    - 情绪识别：通过极端值频率反映市场情绪极端化
    - 动态适应：窗口滑动，实时反映市场情绪变化
    
    优势：
    - 能捕捉市场极端情绪拐点
    - 适合极端行情和反转捕捉
    - 计算高效，参数简单
    - 与趋势、均值等信号低相关
    
    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列
    - roll_num: 滚动窗口大小，推荐区间[20, 1440]
    - threshold: 极端值阈值，推荐区间[1.0, 3.0]
    """
    df = factor_data.unstack().fillna(method='ffill').fillna(0)
    if df.shape[0] <= roll_num:
        return (df['transformed'].replace([np.inf, -np.inf], 0).fillna(0) * 0).astype(int)
    df_norm = rolling_zscore(df['transformed'], roll_num).fillna(method='ffill').fillna(0)
    result = {}
    from numpy.lib.stride_tricks import sliding_window_view
    for col in df_norm.columns:
        arr = df_norm[col].values
        if len(arr) < roll_num:
            result[col] = pd.Series(np.zeros_like(arr), index=df_norm.index)
            continue
        windows = sliding_window_view(arr, roll_num)
        extreme_arr = fast_extreme_count_window(windows, threshold)
        extreme_full = np.concatenate([np.full(roll_num-1, np.nan), extreme_arr])
        result[col] = pd.Series(extreme_full, index=df_norm.index)
    extreme_count = pd.DataFrame(result, index=df_norm.index)
    signal = extreme_count.fillna(0).astype(int)
    return signal


def create_muster(rolling_sets=None, threshold_sets=None):
    """
    生成extreme_count_signal的参数组合
    
    参数：
    - rolling_sets: 滚动窗口大小集合
    - threshold_sets: 极端值阈值集合
    
    返回：
    - muster: Function对象列表
    """
    rolling_sets = rolling_sets if isinstance(
        rolling_sets, list) else default_rolling_range
    threshold_sets = threshold_sets if isinstance(
        threshold_sets, list) else default_threshold_range
    
    muster = []
    for roll_num in rolling_sets:
        for threshold in threshold_sets:
            muster.append(
                Function(function=extreme_count_signal,
                         name='extreme_count_signal',
                         params={
                             'roll_num': int(roll_num),
                             'threshold': float(threshold)
                         }))
    return muster 
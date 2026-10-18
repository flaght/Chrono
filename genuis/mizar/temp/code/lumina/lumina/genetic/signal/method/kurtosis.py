import numpy as np
import pandas as pd
from lumina.genetic.rolling import rolling_zscore
from numba import njit
from lumina.genetic.signal.method.env import Function

@njit
def fast_kurtosis_window(arr):
    n = arr.shape[0]
    out = np.empty(n)
    for i in range(n):
        x = arr[i]
        m = np.mean(x)
        s2 = np.mean((x - m) ** 2)
        s4 = np.mean((x - m) ** 4)
        if s2 == 0:
            out[i] = 0
        else:
            out[i] = s4 / (s2 ** 2) - 3  # Fisher定义，与scipy.stats.kurtosis默认一致
    return out

default_rolling_range = [x for x in range(20, 60, 5)]
default_threshold_range = [round(x, 2) for x in list(np.arange(0.1, 2.1, 0.05))]


def kurtosis_signal(factor_data: pd.DataFrame,
                   roll_num: int = 20,
                   threshold: float = 0.0) -> pd.Series:
    """
    峰度信号模型 - 基于滚动窗口峰度的极端波动检测
    
    策略解释：
    基于统计学中的峰度指标，衡量因子值分布的尾部厚度。通过计算滚动窗口内的峰度，识别市场中极端波动或极端平稳的状态。当峰度大于阈值时，认为市场存在极端波动，产生做空信号；当峰度小于负阈值时，认为市场极端平稳，产生做多信号。
    
    核心思想：
    - 峰度度量：利用滚动窗口内的峰度衡量分布的厚尾特征
    - 极端波动识别：通过峰度的高低识别市场极端风险或极端稳定
    - 动态适应：窗口滑动，实时反映市场波动结构
    
    优势：
    - 能捕捉市场极端风险和黑天鹅事件
    - 适合风险管理和极端行情捕捉
    - 计算高效，参数简单
    - 与均值、动量等信号低相关
    
    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列
    - roll_num: 滚动窗口大小，推荐区间[20, 1440]
    - threshold: 峰度阈值，推荐区间[0.1, 2.0]
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
        kurt_arr = fast_kurtosis_window(windows)
        kurt_full = np.concatenate([np.full(roll_num-1, np.nan), kurt_arr])
        result[col] = pd.Series(kurt_full, index=df_norm.index)
    kurt = pd.DataFrame(result, index=df_norm.index)
    signal = ((kurt < -threshold).astype(int) - (kurt > threshold).astype(int))
    return signal


def create_muster(rolling_sets=None, threshold_sets=None):
    """
    生成kurtosis_signal的参数组合
    
    参数：
    - rolling_sets: 滚动窗口大小集合
    - threshold_sets: 峰度阈值集合
    
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
                Function(function=kurtosis_signal,
                         name='kurtosis_signal',
                         params={
                             'roll_num': int(roll_num),
                             'threshold': float(threshold)
                         }))
    return muster 
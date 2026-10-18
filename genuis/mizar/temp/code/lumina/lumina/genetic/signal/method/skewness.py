import numpy as np
import pandas as pd
from lumina.genetic.rolling import rolling_zscore
from numba import njit
from lumina.genetic.signal.method.env import Function

@njit
def fast_skewness_window(arr):
    n = arr.shape[0]
    out = np.empty(n)
    for i in range(n):
        x = arr[i]
        m = np.mean(x)
        s = np.std(x)
        if s == 0:
            out[i] = 0
        else:
            out[i] = np.mean(((x - m) / s) ** 3)
    return out

default_rolling_range = [x for x in range(20, 60, 5)]
default_threshold_range = [round(x, 2) for x in list(np.arange(0.1, 2.1, 0.05))]


def skewness_signal(factor_data: pd.DataFrame,
                   roll_num: int = 20,
                   threshold: float = 0.0) -> pd.Series:
    """
    偏度信号模型 - 基于滚动窗口偏度的分布不对称性检测
    
    策略解释：
    基于统计学中的偏度指标，衡量因子值分布的对称性。通过计算滚动窗口内的偏度，识别市场中正偏或负偏的极端行为。当偏度大于阈值时，认为市场存在极端正偏，可能出现持续上涨，产生做多信号；当偏度小于负阈值时，认为市场存在极端负偏，可能出现持续下跌，产生做空信号。
    
    核心思想：
    - 偏度度量：利用滚动窗口内的偏度衡量分布的非对称性
    - 极端行为识别：通过偏度的正负识别市场极端情绪
    - 动态适应：窗口滑动，实时反映市场结构变化
    
    优势：
    - 能捕捉市场极端情绪和结构性变化
    - 适合捕捉趋势启动或极端反转时机
    - 计算高效，参数简单
    - 与均值、波动率等信号低相关
    
    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列
    - roll_num: 滚动窗口大小，推荐区间[20, 1440]
    - threshold: 偏度阈值，控制信号敏感度，推荐区间[0.1, 2.0]
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
        skew_arr = fast_skewness_window(windows)
        skew_full = np.concatenate([np.full(roll_num-1, np.nan), skew_arr])
        result[col] = pd.Series(skew_full, index=df_norm.index)
    skewness = pd.DataFrame(result, index=df_norm.index)
    signal = ((skewness > threshold).astype(int) - (skewness < -threshold).astype(int))
    return signal


def create_muster(rolling_sets=None, threshold_sets=None):
    """
    生成skewness_signal的参数组合
    
    参数：
    - rolling_sets: 滚动窗口大小集合
    - threshold_sets: 偏度阈值集合
    
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
                Function(function=skewness_signal,
                         name='skewness_signal',
                         params={
                             'roll_num': int(roll_num),
                             'threshold': float(threshold)
                         }))
    return muster 
import numpy as np
import pandas as pd
from lumina.genetic.rolling import rolling_zscore
from numba import njit
from lumina.genetic.signal.method.env import Function

@njit
def nanmedian_1d(x):
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    x = np.sort(x)
    n = x.size
    if n % 2 == 1:
        return x[n // 2]
    else:
        return 0.5 * (x[n // 2 - 1] + x[n // 2])

@njit
def siegelslopes_end(x, x_idx):
    n = len(x)
    slopes = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                slopes[i, j] = np.nan
            else:
                slopes[i, j] = (x[j] - x[i]) / (x_idx[j] - x_idx[i])
    med_slopes = np.empty(n)
    for i in range(n):
        med_slopes[i] = nanmedian_1d(slopes[i])
    slope = nanmedian_1d(med_slopes)
    intercepts = x - slope * x_idx
    intercept = nanmedian_1d(intercepts)
    return intercept + slope * (n - 1)

default_rolling_range = [x for x in range(20, 60, 5)]
default_threshold_range = [round(x, 2) for x in list(np.arange(0.05, 0.51, 0.01))]


def icu_signal(factor_data: pd.DataFrame,
               roll_num: int = 20,
               threshold: float = 0.0) -> pd.Series:
    """
    ICU信号模型 - 基于Siegel斜率均线的趋势信号

    策略解释：
    基于Siegel repeated median回归的窗口均线，捕捉价格的稳健趋势。窗口内用鲁棒回归拟合，取回归终点作为信号。适合抗噪声、抗异常点的趋势捕捉。

    核心思想：
    - 鲁棒回归：窗口内用Siegel repeated median回归
    - 趋势终点：用回归终点值作为信号
    - 动态适应：窗口滑动，实时反映趋势变化

    优势：
    - 鲁棒性强，抗极端值
    - 能捕捉平滑趋势
    - 适合高频/低频多场景
    - 计算高效（Numba加速）

    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列
    - roll_num: 滚动窗口大小，推荐区间[20, 1440]
    - threshold: 信号阈值（可选，通常不需要），推荐区间[0.05, 0.5]
    """
    df = factor_data.unstack().fillna(method='ffill').fillna(0)
    if df.shape[0] <= roll_num:
        return (df['transformed'].replace([np.inf, -np.inf], 0).fillna(0) * 0).astype(int)
    df_norm = rolling_zscore(df['transformed'], roll_num).fillna(method='ffill').fillna(0)
    result = {}
    from numpy.lib.stride_tricks import sliding_window_view
    x_idx = np.arange(roll_num)
    for col in df_norm.columns:
        arr = df_norm[col].values
        if len(arr) < roll_num:
            result[col] = pd.Series(np.zeros_like(arr), index=df_norm.index)
            continue
        windows = sliding_window_view(arr, roll_num)
        icu_arr = np.empty(windows.shape[0])
        for i in range(windows.shape[0]):
            icu_arr[i] = siegelslopes_end(windows[i], x_idx)
        icu_full = np.concatenate([np.full(roll_num-1, np.nan), icu_arr])
        result[col] = pd.Series(icu_full, index=df_norm.index)
    icu_df = pd.DataFrame(result, index=df_norm.index)
    signal = ((icu_df > threshold).astype(int) - (icu_df < -threshold).astype(int))
    return signal

def create_muster(rolling_sets=None, threshold_sets=None):
    """
    生成icu_signal的参数组合
    
    参数：
    - rolling_sets: 滚动窗口大小集合
    - threshold_sets: 信号阈值集合
    
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
                Function(function=icu_signal,
                         name='icu_signal',
                         params={
                             'roll_num': int(roll_num),
                             'threshold': float(threshold)
                         }))
    return muster 
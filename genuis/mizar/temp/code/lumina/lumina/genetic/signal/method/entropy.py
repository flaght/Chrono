import numpy as np
import pandas as pd
from lumina.genetic.rolling import rolling_zscore
from numba import njit
from lumina.genetic.signal.method.env import Function

default_rolling_range = [x for x in range(20, 60, 5)]
default_threshold_range = [
    round(x, 2) for x in list(np.arange(0.2, 0.7, 0.05))
]
default_bins_range = [x for x in range(5, 10, 1)]


@njit
def fast_entropy_window(arr, bins):
    n = arr.shape[0]
    out = np.empty(n)
    for i in range(n):
        x = arr[i]
        hist = np.histogram(x, bins=bins)[0]  # bins可以为int或array
        total = np.sum(hist)
        if total > 0:
            prob = hist / total
            prob = prob[prob > 0]
            out[i] = -np.sum(prob * np.log(prob + 1e-8))
        else:
            out[i] = 0
    return out


def entropy_signal(factor_data: pd.DataFrame,
                   roll_num: int = 20,
                   threshold: float = 0.0,
                   bins: int = 10) -> pd.Series:
    """
    熵信号模型 - 基于信息熵的混乱度检测策略
    
    策略解释：
    基于信息熵理论的信号策略，通过计算因子值序列的信息熵来检测市场混乱度。
    当熵值较低时表示市场有序，产生趋势跟随信号；当熵值较高时表示市场混乱，
    产生反转信号，体现了"混乱度检测"的自适应交易理念。
    
    核心思想：
    - 信息熵：计算价格序列的信息熵作为混乱度指标
    - 有序检测：低熵值表示市场有序，适合趋势策略
    - 混乱检测：高熵值表示市场混乱，适合反转策略
    - 自适应：根据市场状态自动调整策略
    - 分箱参数：bins 可调，控制熵计算的分辨率，推荐区间[5, 20]
    
    优势：
    - 混乱度检测：能够有效识别市场的有序和混乱状态
    - 自适应策略：根据市场状态自动调整交易策略
    - 高效计算：使用向量化操作，避免循环
    - 低相关性：与传统的价格指标形成互补
    
    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列
    - roll_num: 滚动窗口大小，用于计算信息熵
    - threshold: 熵阈值，控制信号生成的敏感度
    - bins: 分箱数，控制熵计算精度，推荐区间[5, 20]
    """
    df = factor_data.unstack().fillna(method='ffill').fillna(0)
    if df.shape[0] <= roll_num:
        return (df['transformed'].replace([np.inf, -np.inf], 0).fillna(0) *
                0).astype(int)
    df_norm = rolling_zscore(df['transformed'],
                             roll_num).fillna(method='ffill').fillna(0)
    if isinstance(bins, int):
        bins_arr = np.linspace(-3, 3, bins)
    elif isinstance(bins, float):
        bins_arr = np.linspace(-3, 3, int(bins))
    else:
        bins_arr = bins
    result = {}
    from numpy.lib.stride_tricks import sliding_window_view
    for col in df_norm.columns:
        arr = df_norm[col].values
        if len(arr) < roll_num:
            result[col] = pd.Series(np.zeros_like(arr), index=df_norm.index)
            continue
        windows = sliding_window_view(arr, roll_num)
        entropy_arr = fast_entropy_window(windows, bins_arr)
        entropy_full = np.concatenate(
            [np.full(roll_num - 1, np.nan), entropy_arr])
        result[col] = pd.Series(entropy_full, index=df_norm.index)
    entropy_values = pd.DataFrame(result, index=df_norm.index)
    entropy_momentum = entropy_values.diff(roll_num // 2).fillna(0)
    entropy_normalized = (entropy_values - entropy_values.rolling(
        roll_num, min_periods=1).mean()) / (
            entropy_values.rolling(roll_num, min_periods=1).std() + 1e-8)
    trend_strength = df_norm.diff(roll_num // 2).rolling(roll_num // 2,
                                                         min_periods=1).mean()
    low_entropy = entropy_normalized < -threshold
    high_entropy = entropy_normalized > threshold
    trend_signal = ((low_entropy & (trend_strength > 0)).astype(int) -
                    (low_entropy & (trend_strength < 0)).astype(int))
    reversal_signal = ((high_entropy & (entropy_momentum > 0)).astype(int) -
                       (high_entropy & (entropy_momentum < 0)).astype(int))
    signal = trend_signal + reversal_signal
    return signal


def create_muster(rolling_sets=None, threshold_sets=None, bins_sets=None):
    """
    生成entropy_signal的参数组合
    
    参数：
    - rolling_sets: 滚动窗口大小集合
    - threshold_sets: 熵阈值集合
    - bins_sets: 分箱数集合
    
    返回：
    - muster: Function对象列表
    """
    rolling_sets = rolling_sets if isinstance(rolling_sets,
                                              list) else default_rolling_range
    threshold_sets = threshold_sets if isinstance(
        threshold_sets, list) else default_threshold_range
    bins_sets = bins_sets if isinstance(bins_sets,
                                        list) else default_bins_range

    muster = []
    for roll_num in rolling_sets:
        for threshold in threshold_sets:
            for bins in bins_sets:
                muster.append(
                    Function(function=entropy_signal,
                             name='entropy_signal',
                             params={
                                 'roll_num': int(roll_num),
                                 'threshold': float(threshold),
                                 'bins': int(bins)
                             }))
    return muster

import numpy as np
import pandas as pd
from lumina.genetic.rolling import rolling_zscore
from numba import njit
from lumina.genetic.signal.method.env import Function

@njit
def fast_regression_window(arr):
    n = arr.shape[0]
    win = arr.shape[1]
    slopes = np.empty(n)
    r2s = np.empty(n)
    x_vals = np.arange(win)
    x_mean = np.mean(x_vals)
    x_demean = x_vals - x_mean
    x_var = np.sum(x_demean ** 2)
    for i in range(n):
        y = arr[i]
        y_mean = np.mean(y)
        y_demean = y - y_mean
        numerator = np.sum(x_demean * y_demean)
        slope = numerator / (x_var + 1e-8)
        intercept = y_mean - slope * x_mean
        y_pred = slope * x_vals + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8)) if ss_tot > 0 else 0
        slopes[i] = slope
        r2s[i] = r2
    return slopes, r2s

default_rolling_range = [x for x in range(20, 60, 5)]
default_threshold_range = [round(x, 2) for x in list(np.arange(0.05, 1.01, 0.05))]


def regression_signal(factor_data: pd.DataFrame,
                     roll_num: int = 20,
                     threshold: float = 0.0) -> pd.Series:
    """
    回归信号模型 - 基于线性回归的趋势预测策略
    
    策略解释：
    基于线性回归理论的信号策略，通过拟合因子值的时间序列来预测未来趋势。
    当回归斜率显著为正时产生做多信号，当回归斜率显著为负时产生做空信号，
    体现了"趋势预测"的前瞻性交易理念。
    
    核心思想：
    - 线性拟合：使用滚动窗口进行线性回归拟合
    - 斜率预测：通过回归斜率预测未来趋势方向
    - 拟合优度：结合R²值评估预测的可靠性
    - 前瞻性：基于历史数据预测未来趋势
    
    优势：
    - 趋势预测：能够前瞻性地预测趋势方向
    - 统计基础：基于严谨的线性回归理论
    - 高效计算：使用向量化操作，避免循环
    - 低相关性：与滞后指标形成互补
    
    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列
    - roll_num: 滚动窗口大小，用于线性回归拟合，推荐区间[20, 1440]
    - threshold: 斜率阈值，控制信号生成的敏感度，推荐区间[0.05, 1.0]
    """
    df = factor_data.unstack().fillna(method='ffill').fillna(0)
    if df.shape[0] <= roll_num:
        return (df['transformed'].replace([np.inf, -np.inf], 0).fillna(0) * 0).astype(int)
    df_norm = rolling_zscore(df['transformed'], roll_num).fillna(method='ffill').fillna(0)
    result_slope = {}
    result_r2 = {}
    from numpy.lib.stride_tricks import sliding_window_view
    for col in df_norm.columns:
        arr = df_norm[col].values
        if len(arr) < roll_num:
            result_slope[col] = pd.Series(np.zeros_like(arr), index=df_norm.index)
            result_r2[col] = pd.Series(np.zeros_like(arr), index=df_norm.index)
            continue
        windows = sliding_window_view(arr, roll_num)
        slopes, r2s = fast_regression_window(windows)
        slopes_full = np.concatenate([np.full(roll_num-1, np.nan), slopes])
        r2s_full = np.concatenate([np.full(roll_num-1, np.nan), r2s])
        result_slope[col] = pd.Series(slopes_full, index=df_norm.index)
        result_r2[col] = pd.Series(r2s_full, index=df_norm.index)
    slopes_df = pd.DataFrame(result_slope, index=df_norm.index)
    r2_df = pd.DataFrame(result_r2, index=df_norm.index)
    trend_strength = slopes_df * r2_df
    dynamic_threshold = threshold * trend_strength.rolling(roll_num, min_periods=1).std().fillna(threshold)
    signal = ((trend_strength > dynamic_threshold).astype(int) - (trend_strength < -dynamic_threshold).astype(int))
    return signal

def create_muster(rolling_sets=None, threshold_sets=None):
    """
    生成regression_signal的参数组合
    
    参数：
    - rolling_sets: 滚动窗口大小集合
    - threshold_sets: 斜率阈值集合
    
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
                Function(function=regression_signal,
                         name='regression_signal',
                         params={
                             'roll_num': int(roll_num),
                             'threshold': float(threshold)
                         }))
    return muster 
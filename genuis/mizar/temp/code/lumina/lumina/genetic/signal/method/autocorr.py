import numpy as np
import pandas as pd
from lumina.genetic.rolling import rolling_zscore
from lumina.genetic.signal.method.env import Function

default_rolling_range = [x for x in range(20, 60, 5)]
default_threshold_range = [round(x, 3) for x in np.arange(0.05, 0.2, 0.01)]
default_lag_range = [x for x in range(1, 5, 1)]


def autocorr_signal(factor_data: pd.DataFrame,
                    roll_num: int = 20,
                    threshold: float = 0.0,
                    lag: int = 1) -> pd.Series:
    """
    自相关信号模型 - 基于滚动窗口自相关系数的惯性/反转检测
    
    策略解释：
    基于时间序列自相关性原理，衡量因子值序列的惯性或反转特征。通过计算滚动窗口内的自相关系数，识别市场的趋势延续性或反转倾向。当自相关系数大于阈值时，认为存在惯性，产生做多信号；小于负阈值时，认为存在反转，产生做空信号。
    
    核心思想：
    - 自相关度量：利用滚动窗口内的自相关系数衡量序列惯性或反转
    - 趋势/反转识别：通过自相关的正负识别趋势延续或反转
    - 动态适应：窗口滑动，实时反映市场行为变化
    - 滞后参数：lag 可调，支持不同周期的自相关检测
    
    优势：
    - 能捕捉趋势惯性和反转信号
    - 适合趋势跟踪和反转策略
    - 计算高效，参数简单
    - 与均值、动量等信号低相关
    
    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列
    - roll_num: 滚动窗口大小
    - threshold: 自相关阈值，控制信号敏感度
    - lag: 滞后周期，默认为1，推荐区间[1, 5, 10, 30]
    """
    df = factor_data.unstack().fillna(method='ffill').fillna(0)
    if df.shape[0] <= roll_num:
        return (df['transformed'].replace([np.inf, -np.inf], 0).fillna(0) *
                0).astype(int)
    df_norm = rolling_zscore(df['transformed'],
                             roll_num).fillna(method='ffill').fillna(0)
    # 高效向量化自相关计算
    result = []
    for col in df_norm.columns:
        s = df_norm[col]
        autocorr = s.rolling(roll_num,
                             min_periods=roll_num // 2).corr(s.shift(int(lag)))
        result.append(autocorr)
    autocorr_df = pd.concat(result, axis=1)
    autocorr_df.columns = df_norm.columns
    signal = ((autocorr_df > threshold).astype(int) -
              (autocorr_df < -threshold).astype(int))
    return signal  # 直接返回 DataFrame，index=时间，columns=代码


def create_muster(rolling_sets=None, threshold_sets=None, lag_sets=None):
    """
    生成autocorr_signal的参数组合
    
    参数：
    - rolling_sets: 滚动窗口大小集合
    - threshold_sets: 自相关阈值集合
    - lag_sets: 滞后周期集合
    
    返回：
    - muster: Function对象列表
    """
    rolling_sets = rolling_sets if isinstance(rolling_sets,
                                              list) else default_rolling_range
    threshold_sets = threshold_sets if isinstance(
        threshold_sets, list) else default_threshold_range
    lag_sets = lag_sets if isinstance(lag_sets, list) else default_lag_range

    muster = []
    for roll_num in rolling_sets:
        for threshold in threshold_sets:
            for lag in lag_sets:
                muster.append(
                    Function(function=autocorr_signal,
                             name='autocorr_signal',
                             params={
                                 'roll_num': int(roll_num),
                                 'threshold': float(threshold),
                                 'lag': int(lag)
                             }))
    return muster

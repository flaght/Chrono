import numpy as np
import pandas as pd
from lumina.genetic.signal.method.env import Function

# 滚动窗口参数范围
default_rolling_range = [x for x in range(20, 60, 5)]

# 分位数阈值参数范围 (我们将使用一个阈值，另一个通过 1-threshold 计算)
default_quantile_range = [round(x, 2) for x in np.arange(0.80, 1.0, 0.05)]


def rollrank_signal(factor_data: pd.DataFrame,
                        roll_num: int = 24,
                        threshold: float = 0.9) -> pd.Series:
    """
    滚动排名信号模型 - 基于历史数据排名的非参数策略

    策略解释：
    该策略不关注因子值的绝对大小，而是关注其在近期历史数据中的相对位置（排名）。
    它在每个时间点上，回顾过去一段时间（滚动窗口），计算当前因子值在该窗口内的排名，
    并将其标准化。当标准化后的排名超过某个高分位数时，产生做多信号；低于某个
    低分位数时，产生做空信号。

    核心思想：
    - 相对强弱：信号基于因子在历史窗口中的相对排名，而非绝对值。
    - 非参数性：不依赖于数据的特定分布假设，对异常值不敏感。
    - 动态阈值：买卖阈值（分位数）是根据每个窗口内的数据动态计算的。

    优势：
    - 稳健性：由于使用排名，对极端的因子值（outliers）有很强的鲁棒性。
    - 自适应：信号阈值随市场环境（窗口内数据分布）的变化而自适应调整。
    - 普适性：适用于各种不同量纲和分布的因子数据。

    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列。
    - roll_num: 滚动窗口大小。
    - threshold: 正方向分位数阈值，负方向将使用 (1 - threshold)。
    """
    # 数据预处理：解栈并填充缺失值
    df = factor_data.unstack().fillna(method='ffill').fillna(0)

    # 提取核心因子序列
    series = df['transformed']

    if series.shape[0] > roll_num:
        # --- 核心策略实现 ---

        # 为了匹配框架，从一个参数生成两个分位数
        negative_quantile = 1 - threshold

        # 定义一个函数，用于处理每个滚动窗口
        def discretize_window(window_data: np.ndarray) -> int:
            # rolling().apply() 传递的是 numpy array (raw=True)
            # 我们需要先检查NaN
            if np.isnan(window_data).any():
                return np.nan

            # 将numpy array转为Series以使用.rank()和.quantile()
            window_series = pd.Series(window_data)

            # 在窗口内计算排名和标准化值
            rank_data = window_series.rank()

            # 防止 rank_data.sum() 为 0 的情况
            rank_sum = rank_data.sum()
            if rank_sum == 0:
                return 0  # 或者返回 np.nan

            standardized = (rank_data / rank_sum) - 0.5

            # 计算当前窗口的分位数阈值
            pos_threshold = standardized.quantile(threshold)
            neg_threshold = standardized.quantile(threshold)

            # 对窗口内的最后一个值（即当前点）进行离散化
            current_standardized = standardized.iloc[-1]
            if current_standardized >= pos_threshold:
                return 1
            elif current_standardized <= neg_threshold:
                return -1
            else:
                return 0

        # 使用 rolling().apply() 来替代 for 循环，实现向量化
        # raw=True 性能更好，传递的是 numpy 数组
        signal = series.rolling(window=roll_num).apply(discretize_window,
                                                       raw=True)

        # apply之后可能会有NaN，需要填充
        signal = signal.fillna(0).astype(int)

    else:
        # 数据不足时返回零信号
        signal = (series.replace(np.inf, 0).replace(-np.inf, 0).fillna(0) *
                  0).astype(int)

    return signal


def create_muster(rolling_sets=None, threshold_sets=None):
    """
    生成rolling_rank_signal的参数组合
    
    参数：
    - rolling_sets: 滚动窗口大小集合。
    - threshold_sets: 正方向分位数阈值集合。
    
    返回：
    - muster: Function对象列表。
    """
    rolling_sets = rolling_sets if isinstance(rolling_sets,
                                              list) else default_rolling_range
    threshold_sets = threshold_sets if isinstance(
        threshold_sets, list) else default_quantile_range

    muster = []
    for roll_num in rolling_sets:
        for threshold in threshold_sets:
            # 确保阈值在合理范围内
            if threshold <= 0.5 or threshold >= 1.0:
                continue

            muster.append(
                Function(function=rollrank_signal,
                         name='rollrank_signal',
                         params={
                             'roll_num': int(roll_num),
                             'threshold': float(threshold)
                         }))
    return muster

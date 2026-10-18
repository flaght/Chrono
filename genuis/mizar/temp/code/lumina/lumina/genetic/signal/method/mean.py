import numpy as np
import pandas as pd
from lumina.genetic.rolling import rolling_zscore
from lumina.genetic.signal.method.env import Function

default_rolling_range = [x for x in range(20, 60, 5)]
default_threshold_range = [
    round(x, 2) for x in list(np.arange(0.05, 0.5, 0.05))
]


def mean_signal(factor_data: pd.DataFrame,
                roll_num: int = 20,
                threshold: float = 0.0) -> pd.Series:
    """
    均值回归信号模型 - 基于滚动均值的趋势策略
    
    策略解释：
    基于均值回归理论的信号策略，通过比较当前因子值与滚动窗口均值的偏离程度
    来生成交易信号。当因子值显著高于历史均值时产生做空信号，当因子值显著
    低于历史均值时产生做多信号，体现了"价格终将回归均值"的投资理念。
    
    核心思想：
    - 均值回归：因子值偏离均值后终将回归
    - 滚动均值：使用滚动窗口计算动态均值基准
    - 偏离程度：通过阈值控制信号生成的敏感度
    - 趋势捕捉：捕捉因子值的均值回归机会
    
    优势：
    - 理论基础：基于经典的均值回归理论
    - 简单有效：逻辑清晰，易于理解和实现
    - 风险控制：通过阈值控制避免过度交易
    - 适应性：滚动窗口能够适应市场环境变化
    
    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列
    - roll_num: 滚动窗口大小，用于计算均值基准，推荐区间[20, 1440]
    - threshold: 偏离阈值，控制信号生成的敏感度，推荐区间[0.05, 1.0]
    """
    # 数据预处理：解栈并填充缺失值
    df = factor_data.unstack().fillna(method='ffill').fillna(0)

    if df.shape[0] > roll_num:
        # 策略实现：标准化因子值以消除量纲影响
        df_current = rolling_zscore(df['transformed'],
                                    roll_num).fillna(method='ffill').fillna(0)

        # 策略实现：计算滚动窗口均值作为回归基准
        rolling_mean = df.rolling(roll_num).mean()

        # 策略实现：生成均值回归信号，偏离阈值时产生交易信号
        #signal = (df > threshold).astype(int) - (df < -threshold).astype(int)
        # 均值回归信号
        signal = (df_current < rolling_mean - threshold).astype(int) - (
            df_current > rolling_mean + threshold).astype(int)
    else:
        # 数据不足时返回零信号
        signal = (df['transformed'].replace(np.inf, 0).replace(
            -np.inf, 0).fillna(0) * 0).astype(int)

    return signal


def create_muster(rolling_sets=None, threshold_sets=None):
    """
    生成mean_signal的参数组合
    
    参数：
    - rolling_sets: 滚动窗口大小集合
    - threshold_sets: 偏离阈值集合
    
    返回：
    - muster: Function对象列表
    """
    rolling_sets = rolling_sets if isinstance(rolling_sets,
                                              list) else default_rolling_range
    threshold_sets = threshold_sets if isinstance(
        threshold_sets, list) else default_threshold_range

    muster = []
    for roll_num in rolling_sets:
        for threshold in threshold_sets:
            muster.append(
                Function(function=mean_signal,
                         name='mean_signal',
                         params={
                             'roll_num': int(roll_num),
                             'threshold': float(threshold)
                         }))
    return muster

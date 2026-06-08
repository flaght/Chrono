import numpy as np
import pandas as pd
from lumina.genetic.rolling import rolling_zscore
from lumina.genetic.signal.method.env import Function

default_rolling_range = [x for x in range(20, 60, 5)]
default_threshold_range = [round(x, 3) for x in np.arange(0.01, 0.05, 0.01)]


def adaptive_signal(factor_data: pd.DataFrame,
                    roll_num: int = 20,
                    threshold: float = 0.0) -> pd.Series:
    """
    自适应分位数信号模型 - 动态阈值调整策略
    
    策略解释：
    基于波动率自适应的分位数策略，通过动态调整分位数阈值来适应市场波动性变化。
    当市场波动率较高时，自动扩大分位数区间以降低信号敏感度；当波动率较低时，
    缩小分位数区间以增强信号敏感度。这种自适应机制能够有效应对不同市场环境。
    
    核心思想：
    - 波动率标准化：使用滚动标准差对因子值进行标准化
    - 动态分位数：根据标准化后的数据动态计算上下分位数阈值
    - 自适应调整：高波动率时扩大阈值区间，低波动率时缩小阈值区间
    
    优势：
    - 环境适应：自动适应不同市场波动环境
    - 风险控制：高波动率时降低信号频率
    - 机会捕捉：低波动率时增强信号敏感度
    - 稳健性：基于分位数的非参数方法
    
    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列
    - roll_num: 滚动窗口大小
    - threshold: 分位数阈值，控制信号生成的敏感度
    """
    # 数据预处理：解栈并填充缺失值
    df = factor_data.unstack().fillna(method='ffill').fillna(0)

    if df.shape[0] > roll_num:
        # 策略实现：标准化因子值以消除量纲影响
        df = rolling_zscore(df['transformed'],
                            roll_num).fillna(method='ffill').fillna(0)

        # 策略实现：计算滚动波动率作为标准化因子
        std = df.rolling(roll_num).std()

        # 策略实现：波动率标准化，实现自适应调整
        scaled = df / std

        # 策略实现：计算动态分位数阈值，实现自适应信号生成
        upper = scaled.rolling(roll_num).quantile(1 - threshold)  # 上分位数
        lower = scaled.rolling(roll_num).quantile(threshold)  # 下分位数

        # 策略实现：生成多空信号，突破分位数阈值时产生信号
        signal = (scaled > upper).astype(int) - (scaled < lower)
    else:
        # 数据不足时返回零信号
        signal = (df['transformed'].replace(np.inf, 0).replace(
            -np.inf, 0).fillna(0) * 0).astype(int)

    return signal


def create_muster(rolling_sets=None, threshold_sets=None):
    """
    生成adaptive_signal的参数组合
    
    参数：
    - rolling_sets: 滚动窗口大小集合
    - threshold_sets: 分位数阈值集合
    
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
                Function(function=adaptive_signal,
                         name='adaptive_signal',
                         params={
                             'roll_num': int(roll_num),
                             'threshold': float(threshold)
                         }))
    return muster

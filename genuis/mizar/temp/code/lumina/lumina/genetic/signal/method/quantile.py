import numpy as np
import pandas as pd
from lumina.genetic.rolling import rolling_zscore
from lumina.genetic.signal.method.env import Function


default_rolling_range = [x for x in range(20, 60, 5)]
default_threshold_range = [round(x, 2) for x in list(np.arange(0.05, 0.5, 0.05))]


def quantile_signal(factor_data: pd.DataFrame,
                    roll_num: int = 20,
                    threshold: float = 0.7) -> pd.Series:
    """
    分位数信号模型 - 基于超买超卖的区间突破策略
    
    策略解释：
    基于分位数突破的信号策略，参考RSI超买超卖指标的设计理念。通过计算滚动窗口
    内的分位数阈值来识别极值区域，当因子值突破上分位数时产生做空信号（超买），
    当因子值突破下分位数时产生做多信号（超卖），体现了"物极必反"的反转交易理念。
    
    核心思想：
    - 超买超卖：识别因子值的极值区域
    - 分位数阈值：使用滚动分位数作为动态阈值
    - 区间突破：突破分位数边界时产生反转信号
    - 反转交易：在极值区域进行反向操作
    
    优势：
    - 极值识别：能够有效识别超买超卖区域
    - 反转捕捉：捕捉价格反转的机会
    - 动态阈值：分位数阈值随市场环境动态调整
    - 风险控制：避免在极值区域追涨杀跌
    
    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列
    - roll_num: 滚动窗口大小，用于计算分位数阈值，推荐区间[20, 1440]
    - threshold: 分位数阈值，控制超买超卖区域的敏感度，推荐区间[0.05, 0.5]
    """
    # 数据预处理：解栈并填充缺失值
    df = factor_data.unstack().fillna(method='ffill').fillna(0)
    
    if df.shape[0] > roll_num:
        # 策略实现：标准化因子值以消除量纲影响
        df = rolling_zscore(df['transformed'],
                            roll_num).fillna(method='ffill').fillna(0)
        
        # 策略实现：计算动态分位数阈值，识别超买超卖区域
        long_threshold: pd.DataFrame = df.rolling(roll_num).quantile(1 - threshold)  # 上分位数（超买线）
        short_threshold: pd.DataFrame = df.rolling(roll_num).quantile(threshold)     # 下分位数（超卖线）
        
        # 策略实现：生成分位数突破信号，极值区域产生反转信号
        # 正确的均值回归逻辑
        signal = (df < short_threshold).astype(int) - (df > long_threshold).astype(int)
    else:
        # 数据不足时返回零信号
        signal = (df['transformed'].replace(np.inf, 0).replace(
            -np.inf, 0).fillna(0) * 0).astype(int)

    return signal


def create_muster(rolling_sets=None, threshold_sets=None):
    """
    生成quantile_signal的参数组合
    
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
                Function(function=quantile_signal,
                         name='quantile_signal',
                         params={
                             'roll_num': int(roll_num),
                             'threshold': float(threshold)
                         }))
    return muster

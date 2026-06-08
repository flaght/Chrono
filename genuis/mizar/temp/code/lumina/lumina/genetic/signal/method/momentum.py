import numpy as np
import pandas as pd
from lumina.genetic.rolling import rolling_zscore
from lumina.genetic.signal.method.env import Function


default_rolling_range = [x for x in range(20, 60, 5)]
default_threshold_range = [round(x, 2) for x in list(np.arange(0.05, 0.3, 0.05))]


def momentum_signal(factor_data: pd.DataFrame,
                    roll_num: int = 20,
                    threshold: float = 0.0) -> pd.Series:
    """
    动量信号模型 - 基于价格变化的趋势跟踪策略
    
    策略解释：
    基于动量效应的信号策略，通过捕捉因子值的变化趋势来生成交易信号。
    当因子值呈现上升趋势时产生做多信号，当因子值呈现下降趋势时产生做空信号，
    体现了"趋势延续"的投资理念，即价格运动具有惯性特征。
    
    核心思想：
    - 动量效应：价格变化具有惯性，趋势会延续
    - 变化率计算：通过差分计算因子值的变化速度
    - 趋势平滑：使用滚动均值平滑短期波动
    - 趋势跟踪：跟随因子值的变化方向进行交易
    
    优势：
    - 趋势捕捉：能够有效捕捉中期趋势机会
    - 惯性利用：利用价格运动的惯性特征
    - 平滑处理：通过均值平滑减少噪音干扰
    - 适应性：滚动窗口能够适应不同市场环境
    
    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列
    - roll_num: 滚动窗口大小，用于计算变化率和均值，推荐区间[20, 1440]
    - threshold: 趋势阈值，控制信号生成的敏感度，推荐区间[0.05, 0.5]
    """
    # 数据预处理：解栈并填充缺失值
    df = factor_data.unstack().fillna(method='ffill').fillna(0)
    
    if df.shape[0] > roll_num:
        # 策略实现：标准化因子值以消除量纲影响
        df = rolling_zscore(df['transformed'],
                            roll_num).fillna(method='ffill').fillna(0)
        
        # 策略实现：计算动量变化率并平滑处理
        df = df.diff(roll_num).rolling(roll_num).mean()
        
        # 策略实现：生成动量信号，趋势突破阈值时产生交易信号
        signal = (df > threshold).astype(int) - (df < -threshold).astype(int)
    else:
        # 数据不足时返回零信号
        signal = (df['transformed'].replace(np.inf, 0).replace(
            -np.inf, 0).fillna(0) * 0).astype(int)

    return signal


def create_muster(rolling_sets=None, threshold_sets=None):
    """
    生成momentum_signal的参数组合
    
    参数：
    - rolling_sets: 滚动窗口大小集合
    - threshold_sets: 趋势阈值集合
    
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
                Function(function=momentum_signal,
                         name='momentum_signal',
                         params={
                             'roll_num': int(roll_num),
                             'threshold': float(threshold)
                         }))
    return muster

import numpy as np
import pandas as pd
from lumina.genetic.rolling import rolling_zscore
from lumina.genetic.signal.method.env import Function


default_rolling_range = [x for x in range(20, 60, 5)]
default_threshold_range = [round(x, 2) for x in list(np.arange(0.05, 1.01, 0.05))]

def simple_signal(factor_data: pd.DataFrame,
                  roll_num: int = 20,
                  threshold: float = 0.0) -> pd.Series:
    """
    简单信号模型 - 基于零轴突破的基础策略
    
    策略解释：
    基于零轴突破的简单信号策略，参考均线金叉死叉的设计理念。通过比较标准化后
    的因子值与零轴（或设定阈值）的关系来生成交易信号。当因子值突破上阈值时
    产生做多信号，当因子值突破下阈值时产生做空信号，体现了"突破即信号"的
    简单有效交易理念。
    
    核心思想：
    - 零轴突破：以零轴为基准的突破策略
    - 标准化处理：消除量纲影响，便于比较
    - 阈值控制：通过阈值调整信号敏感度
    - 简单有效：逻辑清晰，计算简单
    
    优势：
    - 简单直观：逻辑清晰，易于理解和实现
    - 计算高效：计算量小，执行速度快
    - 参数较少：只需要调整阈值参数
    - 基础策略：可作为其他复杂策略的基础
    
    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列
    - roll_num: 滚动窗口大小，用于标准化处理
    - threshold: 突破阈值，控制信号生成的敏感度
    """
    # 数据预处理：解栈并填充缺失值
    df = factor_data.unstack().fillna(method='ffill').fillna(0)
    
    if df.shape[0] > roll_num:
        # 策略实现：标准化因子值以消除量纲影响
        df = rolling_zscore(df['transformed'],
                            roll_num).fillna(method='ffill').fillna(0)
        
        # 策略实现：生成零轴突破信号，突破阈值时产生交易信号
        signal = (df > threshold).astype(int) - (df < -threshold).astype(int)
    else:
        # 数据不足时返回零信号
        signal = (df['transformed'].replace(np.inf, 0).replace(
            -np.inf, 0).fillna(0) * 0).astype(int)

    return signal

def create_muster(rolling_sets=None, threshold_sets=None):
    """
    生成simple_signal的参数组合
    
    参数：
    - rolling_sets: 滚动窗口大小集合
    - threshold_sets: 突破阈值集合
    
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
                Function(function=simple_signal,
                         name='simple_signal',
                         params={
                             'roll_num': int(roll_num),
                             'threshold': float(threshold)
                         }))
    return muster

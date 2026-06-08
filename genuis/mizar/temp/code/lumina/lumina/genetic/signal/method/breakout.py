import numpy as np
import pandas as pd
from lumina.genetic.rolling import rolling_zscore
from lumina.genetic.signal.method.env import Function


default_rolling_range = [x for x in range(20, 60, 5)]
default_threshold_range = [round(x, 2) for x in list(np.arange(1.0, 3.1, 0.1))]


def breakout_signal(factor_data: pd.DataFrame,
                   roll_num: int = 20,
                   threshold: float = 0.0) -> pd.Series:
    """
    突破信号模型 - 基于价格突破的趋势确认策略
    
    策略解释：
    基于价格突破理论的信号策略，通过识别因子值对关键阻力位和支撑位的突破
    来生成交易信号。当因子值突破上轨阻力位时产生做多信号，当因子值突破
    下轨支撑位时产生做空信号，体现了"突破确认趋势"的交易理念。
    
    核心思想：
    - 通道构建：基于滚动窗口构建价格通道
    - 突破确认：识别对通道边界的有效突破
    - 趋势确认：突破信号作为趋势确认的依据
    - 动量增强：突破后的动量增强效应
    
    优势：
    - 趋势确认：能够有效确认趋势的启动和延续
    - 突破捕捉：及时捕捉价格突破的关键时点
    - 高效计算：使用向量化操作，避免循环
    - 低相关性：与均值回归策略形成互补
    
    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列
    - roll_num: 滚动窗口大小，用于构建价格通道
    - threshold: 突破阈值，控制信号生成的敏感度
    - 推荐区间：roll_num [20, 1440]，threshold [1.0, 3.0]
    """
    # 数据预处理：解栈并填充缺失值
    df = factor_data.unstack().fillna(method='ffill').fillna(0)
    
    if df.shape[0] <= roll_num:
        # 数据不足时返回零信号
        return (df['transformed'].replace([np.inf, -np.inf], 0).fillna(0) * 0).astype(int)
    
    # 策略实现：标准化因子值以消除量纲影响
    df_norm = rolling_zscore(df['transformed'], roll_num).fillna(method='ffill').fillna(0)
    
    # 策略实现：构建价格通道（布林带类似）
    rolling_mean = df_norm.rolling(roll_num, min_periods=1).mean()
    rolling_std = df_norm.rolling(roll_num, min_periods=1).std()
    
    # 策略实现：计算通道边界
    upper_band = rolling_mean + threshold * rolling_std
    lower_band = rolling_mean - threshold * rolling_std
    
    # 策略实现：计算突破强度（距离通道边界的标准化距离）
    upper_breakout = (df_norm - upper_band) / (rolling_std + 1e-8)
    lower_breakout = (lower_band - df_norm) / (rolling_std + 1e-8)
    
    # 策略实现：计算突破确认（连续突破的累积效应）
    upper_confirmed = upper_breakout.rolling(3, min_periods=1).sum()
    lower_confirmed = lower_breakout.rolling(3, min_periods=1).sum()
    
    # 策略实现：计算动量增强因子
    momentum_factor = df_norm.diff(roll_num).rolling(roll_num, min_periods=1).mean()
    
    # 策略实现：生成突破信号，结合突破强度和动量确认
    breakout_signal_up = (upper_confirmed > 0) & (momentum_factor > 0)
    breakout_signal_down = (lower_confirmed > 0) & (momentum_factor < 0)
    
    # 策略实现：应用阈值生成多空信号
    signal = (breakout_signal_up.astype(int) - breakout_signal_down.astype(int))
    
    return signal 


def create_muster(rolling_sets=None, threshold_sets=None):
    """
    生成breakout_signal的参数组合
    
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
                Function(function=breakout_signal,
                         name='breakout_signal',
                         params={
                             'roll_num': int(roll_num),
                             'threshold': float(threshold)
                         }))
    return muster 
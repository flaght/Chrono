import numpy as np
import pandas as pd
from lumina.genetic.rolling import rolling_zscore
from lumina.genetic.signal.method.env import Function


default_rolling_range = [x for x in range(20, 60, 5)]
default_threshold_range = [round(x, 2) for x in list(np.arange(0.5, 3.1, 0.1))]


def triple_barrier_signal(factor_data: pd.DataFrame,
                          roll_num: int = 20,
                          threshold: float = 1.0) -> pd.Series:
    """
    三重屏障信号模型 - 基于布林带突破的趋势跟踪策略
    
    策略解释：
    基于三重屏障理论的信号策略，参考布林带和趋势跟踪策略的设计理念。通过计算
    滚动均值和标准差构建上下轨通道，当因子值突破上轨且继续上涨时产生做多信号，
    当因子值突破下轨且继续下跌时产生做空信号，体现了"突破确认"的趋势跟踪理念。
    
    核心思想：
    - 通道突破：基于均值±标准差构建动态通道
    - 趋势确认：突破通道的同时需要趋势方向确认
    - 动量验证：通过差分验证突破的持续性
    - 趋势跟踪：跟随确认的突破方向进行交易
    
    优势：
    - 趋势捕捉：能够有效捕捉趋势性突破机会
    - 假突破过滤：通过趋势确认过滤假突破
    - 动态通道：通道边界随市场波动动态调整
    - 风险控制：在通道内时避免交易，降低噪音
    
    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列
    - roll_num: 滚动窗口大小，用于计算均值和标准差，推荐区间[20, 1440]
    - threshold: 通道倍数，控制通道宽度的敏感度，推荐区间[0.5, 3.0]
    """
    # 数据预处理：解栈并填充缺失值
    df = factor_data.unstack().fillna(method='ffill').fillna(0)
    
    if df.shape[0] > roll_num:
        # 策略实现：标准化因子值以消除量纲影响
        df = rolling_zscore(df['transformed'],
                            roll_num).fillna(method='ffill').fillna(0)
        
        # 策略实现：计算动态通道边界，构建三重屏障
        ma = df.rolling(roll_num).mean()      # 中轨（均值线）
        std = df.rolling(roll_num).std()      # 标准差
        upper = ma + std * threshold          # 上轨（阻力线）
        lower = ma - std * threshold          # 下轨（支撑线）
        
        # 策略实现：生成三重屏障信号，突破确认时产生趋势跟踪信号
        signal = ((df > upper) & (df.diff() > 0)).astype(int) - \
                 ((df < lower) & (df.diff() < 0)).astype(int)
    else:
        # 数据不足时返回零信号
        signal = (df['transformed'].replace(np.inf, 0).replace(
            -np.inf, 0).fillna(0) * 0).astype(int)
    
    return signal


def create_muster(rolling_sets=None, threshold_sets=None):
    """
    生成triple_barrier_signal的参数组合
    
    参数：
    - rolling_sets: 滚动窗口大小集合
    - threshold_sets: 通道倍数集合
    
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
                Function(function=triple_barrier_signal,
                         name='triple_barrier_signal',
                         params={
                             'roll_num': int(roll_num),
                             'threshold': float(threshold)
                         }))
    return muster

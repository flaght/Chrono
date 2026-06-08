import numpy as np
import pandas as pd
from lumina.genetic.rolling import rolling_zscore
from lumina.genetic.signal.method.env import Function

default_rolling_range = [x for x in range(20, 60, 5)]
default_threshold_range = [
    round(x, 2) for x in list(np.arange(0.05, 0.3, 0.05))
]


def oscillator_signal(factor_data: pd.DataFrame,
                      roll_num: int = 20,
                      threshold: float = 0.0) -> pd.Series:
    """
    振荡器信号模型 - 基于价格振荡的周期策略
    
    策略解释：
    基于价格振荡理论的信号策略，通过识别因子值的周期性波动来生成交易信号。
    当因子值处于振荡周期的低点时产生做多信号，当因子值处于振荡周期的高点时
    产生做空信号，体现了"周期轮动"的投资理念。
    
    核心思想：
    - 周期识别：通过滚动窗口识别价格振荡周期
    - 极值检测：识别周期内的局部极值点
    - 振荡强度：计算振荡幅度作为信号强度
    - 周期轮动：在周期低点买入，高点卖出
    
    优势：
    - 周期捕捉：能够有效识别价格振荡周期
    - 极值利用：在周期极值点进行反向操作
    - 高效计算：使用向量化操作，避免循环
    - 低相关性：与趋势策略形成互补
    
    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列
    - roll_num: 滚动窗口大小，用于识别振荡周期，推荐区间[20, 1440]
    - threshold: 振荡阈值，控制信号生成的敏感度，推荐区间[0.05, 0.3]
    """
    # 数据预处理：解栈并填充缺失值
    df = factor_data.unstack().fillna(method='ffill').fillna(0)

    if df.shape[0] <= roll_num:
        # 数据不足时返回零信号
        return (df['transformed'].replace([np.inf, -np.inf], 0).fillna(0) *
                0).astype(int)

    # 策略实现：标准化因子值以消除量纲影响
    df_norm = rolling_zscore(df['transformed'],
                             roll_num).fillna(method='ffill').fillna(0)

    # 策略实现：计算滚动窗口内的局部极值
    rolling_max = df_norm.rolling(roll_num, min_periods=1).max()
    rolling_min = df_norm.rolling(roll_num, min_periods=1).min()

    # 策略实现：计算振荡幅度和中心线
    oscillation_range = rolling_max - rolling_min
    center_line = (rolling_max + rolling_min) / 2

    # 策略实现：计算相对位置（距离中心线的标准化距离）
    relative_position = (df_norm - center_line) / (oscillation_range + 1e-8)

    # 策略实现：计算振荡强度（基于历史振荡幅度的标准化）
    oscillation_intensity = oscillation_range / (
        oscillation_range.rolling(roll_num, min_periods=1).mean() + 1e-8)

    # 策略实现：生成振荡器信号，结合相对位置和振荡强度
    oscillator_value = relative_position * oscillation_intensity

    # 策略实现：应用阈值生成多空信号
    signal = ((oscillator_value < -threshold).astype(int) -
              (oscillator_value > threshold).astype(int))

    return signal


def create_muster(rolling_sets=None, threshold_sets=None):
    """
    生成oscillator_signal的参数组合
    
    参数：
    - rolling_sets: 滚动窗口大小集合
    - threshold_sets: 振荡阈值集合
    
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
                Function(function=oscillator_signal,
                         name='oscillator_signal',
                         params={
                             'roll_num': int(roll_num),
                             'threshold': float(threshold)
                         }))
    return muster

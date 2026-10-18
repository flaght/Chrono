import numpy as np
import pandas as pd
from lumina.genetic.rolling import rolling_zscore
from lumina.genetic.signal.method.env import Function


default_rolling_range = [x for x in range(20, 60, 5)]
default_threshold_range = [round(x, 2) for x in list(np.arange(0.05, 0.2, 0.01))]


def divergence_signal(factor_data: pd.DataFrame,
                     roll_num: int = 20,
                     threshold: float = 0.3) -> pd.Series:
    """
    背离信号模型 - 基于RSI背离检测的反转策略
    
    策略解释：
    基于技术分析中的背离理论，通过检测因子值与滚动RSI之间的背离关系来生成
    交易信号。当因子值创新高而RSI未创新高时产生做空信号（顶背离），当因子值
    创新低而RSI未创新低时产生做多信号（底背离），体现了"背离即反转"的技术
    分析理念。
    
    核心思想：
    - 背离检测：识别价格与指标之间的背离关系
    - RSI计算：使用滚动窗口计算相对强弱指数
    - 极值比较：比较当前极值与历史极值的关系
    - 反转交易：在背离区域进行反向操作
    
    优势：
    - 反转捕捉：能够有效捕捉趋势反转机会
    - 背离识别：识别价格与指标的背离信号
    - 技术分析：基于经典的技术分析理论
    - 风险预警：背离信号往往预示趋势转折
    
    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列
    - roll_num: 滚动窗口大小，用于计算RSI和极值
    - threshold: 背离阈值，控制背离检测的敏感度
    - 推荐区间：roll_num [20, 1440]，threshold [0.05, 0.3]
    """
    # 数据预处理：解栈并填充缺失值
    df = factor_data.unstack().fillna(method='ffill').fillna(0)
    
    if df.shape[0] <= roll_num * 2:
        # 数据不足时返回零信号
        return (df['transformed'].replace([np.inf, -np.inf], 0).fillna(0) * 0).astype(int)
    
    # 策略实现：标准化因子值以消除量纲影响
    df_norm = rolling_zscore(df['transformed'], roll_num).fillna(method='ffill').fillna(0)
    
    # 策略实现：计算滚动RSI指标
    def calculate_rsi(series, window):
        """计算相对强弱指数"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(df_norm, roll_num)
    
    # 策略实现：计算滚动极值
    rolling_max = df_norm.rolling(roll_num, min_periods=1).max()
    rolling_min = df_norm.rolling(roll_num, min_periods=1).min()
    rsi_max = rsi.rolling(roll_num, min_periods=1).max()
    rsi_min = rsi.rolling(roll_num, min_periods=1).min()
    
    # 策略实现：检测背离信号
    # 顶背离：价格创新高但RSI未创新高
    top_divergence = ((df_norm >= rolling_max * (1 - threshold)) & 
                      (rsi < rsi_max * (1 - threshold)) & 
                      (df_norm.diff() > 0))
    
    # 底背离：价格创新低但RSI未创新低
    bottom_divergence = ((df_norm <= rolling_min * (1 + threshold)) & 
                         (rsi > rsi_min * (1 + threshold)) & 
                         (df_norm.diff() < 0))
    
    # 策略实现：生成背离信号，背离确认时产生反转信号
    signal = bottom_divergence.astype(int) - top_divergence.astype(int)
    
    return signal 


def create_muster(rolling_sets=None, threshold_sets=None):
    """
    生成divergence_signal的参数组合
    
    参数：
    - rolling_sets: 滚动窗口大小集合
    - threshold_sets: 背离阈值集合
    
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
                Function(function=divergence_signal,
                         name='divergence_signal',
                         params={
                             'roll_num': int(roll_num),
                             'threshold': float(threshold)
                         }))
    return muster 
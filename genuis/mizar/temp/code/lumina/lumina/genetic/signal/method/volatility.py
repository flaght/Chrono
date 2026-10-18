import numpy as np
import pandas as pd
from lumina.genetic.rolling import rolling_zscore
from lumina.genetic.signal.method.env import Function


default_rolling_range = [x for x in range(20, 60, 5)]
default_threshold_range = [round(x, 2) for x in list(np.arange(0.05, 1.01, 0.05))]


def volatility_signal(factor_data: pd.DataFrame,
                     roll_num: int = 20,
                     threshold: float = 0.0) -> pd.Series:
    """
    波动率信号模型 - 高效实现，适配分钟频数据
    
    策略解释：
    基于波动率调整的动量策略，通过动态波动率阈值来调整信号敏感度。
    当市场波动率较高时，降低信号强度以控制风险；当波动率较低时，
    增强信号强度以捕捉趋势机会。结合了动量策略的趋势跟踪能力和
    波动率策略的风险控制优势。
    
    核心思想：
    - 波动率标准化：使用滚动波动率对动量信号进行标准化
    - 动态阈值：根据历史波动率均值动态调整信号阈值
    - 风险调整：高波动率环境下自动降低信号敏感度
    
    优势：
    - 高效计算：向量化操作，避免循环
    - 风险调整：动态波动率阈值
    - 趋势捕捉：结合动量策略
    - 分钟频优化：减少不必要的计算
    
    参数：
    - factor_data: 因子数据DataFrame，包含'transformed'列
    - roll_num: 滚动窗口大小（分钟数），推荐区间[20, 1440]
    - threshold: 信号阈值，推荐区间[0.05, 1.0]
    """
    # 数据预处理：解栈并填充缺失值
    df = factor_data.unstack().fillna(method='ffill').fillna(0)
    
    if df.shape[0] <= roll_num:
        # 数据不足时返回零信号
        return (df['transformed'].replace([np.inf, -np.inf], 0).fillna(0) * 0).astype(int)
    
    # 策略实现：标准化因子值以消除量纲影响
    df_norm = rolling_zscore(df['transformed'], roll_num).fillna(method='ffill').fillna(0)
    
    # 策略实现：计算滚动波动率作为风险调整因子
    volatility = df_norm.rolling(roll_num, min_periods=1).std().fillna(0)
    
    # 策略实现：计算动量信号（价格变化率）
    momentum = df_norm.diff(roll_num).fillna(0)
    
    # 策略实现：波动率标准化动量信号，实现风险调整
    vol_adjusted = momentum / (volatility + 1e-8)
    
    # 策略实现：计算动态阈值，基于历史波动率均值自适应调整
    dynamic_thresh = threshold * volatility.rolling(roll_num, min_periods=1).mean().fillna(threshold)
    
    # 策略实现：生成多空信号，高波动率时降低敏感度
    signal = ((vol_adjusted > dynamic_thresh).astype(int) - 
              (vol_adjusted < -dynamic_thresh).astype(int))
    
    return signal 


def create_muster(rolling_sets=None, threshold_sets=None):
    """
    生成volatility_signal的参数组合
    
    参数：
    - rolling_sets: 滚动窗口大小集合
    - threshold_sets: 信号阈值集合
    
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
                Function(function=volatility_signal,
                         name='volatility_signal',
                         params={
                             'roll_num': int(roll_num),
                             'threshold': float(threshold)
                         }))
    return muster 
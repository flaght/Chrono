import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Config:
    # 阈值模式
    threshold_mode: str = 'fixed'  # 'fixed', 'score_diff', 'score_strength', 'position_risk'

    # 固定阈值参数
    threshold: float = 0.5
    
    # Score 差异阈值参数
    base_threshold: float = 0.5
    threshold_k: float = 0.3
    threshold_min: float = 0.2
    threshold_max: float = 0.8

    # 成本参数
    base_cost: float = 0.0001
    cost_multiplier: float = 2000.0

    # 成本模式
    cost_mode: str = 'fixed'  # 'fixed' 或 'proportional'
    
    def __post_init__(self):
        """计算 min_confidence"""
        self.min_confidence = self.cost_multiplier * self.base_cost
        
        
def signal_fixed(action: np.ndarray, config: Config) -> Tuple[float, float, int]:
    """
    固定阈值的信号转换
    
    Args:
        action: [long_score, short_score], shape=(2,)
        config: 信号配置
    
    Returns:
        signal: 交易信号，∈ [-1, +1]
        confidence: 置信度，∈ [0, 1]
        direction: 方向，∈ {-1, 0, +1}
    """
    
    long_score, short_score = action[0], action[1]
    threshold = config.threshold
    min_confidence = config.min_confidence
    
    # 判断方向和强度
    if long_score > threshold and long_score > short_score:
        direction = +1
        confidence = long_score
    elif short_score > threshold and short_score > long_score:
        direction = -1
        confidence = short_score
    else:
        direction = 0
        confidence = 0.0
    
    # 成本硬约束
    if confidence > 0 and confidence < min_confidence:
        direction = 0
        confidence = 0.0
    
    signal = direction * confidence
    return signal, confidence, direction
        
def signal_score_diff(action: np.ndarray, config: Config) -> Tuple[float, float, int]:
    """
    基于 Score 差异的自适应阈值
    
    Args:
        action: [long_score, short_score], shape=(2,)
        config: 信号配置
    
    Returns:
        signal: 交易信号，∈ [-1, +1]
        confidence: 置信度，∈ [0, 1]
        direction: 方向，∈ {-1, 0, +1}
    """
    long_score, short_score = action[0], action[1]
    
    # 计算自适应阈值
    diff = abs(long_score - short_score)
    threshold = config.base_threshold - config.threshold_k * diff
    threshold = max(config.threshold_min, min(config.threshold_max, threshold))

    min_confidence = config.min_confidence
    
    # 判断方向和强度
    if long_score > threshold and long_score > short_score:
        direction = +1
        confidence = long_score
    elif short_score > threshold and short_score > long_score:
        direction = -1
        confidence = short_score
    else:
        direction = 0
        confidence = 0.0
    
    # 成本硬约束
    if confidence > 0 and confidence < min_confidence:
        direction = 0
        confidence = 0.0
    
    signal = direction * confidence
    return signal, confidence, direction
    
def signal_score_strength(action: np.ndarray, config: Config) -> Tuple[float, float, int]:
    """
    基于 Score 强度的自适应阈值
    
    Args:
        action: [long_score, short_score], shape=(2,)
        config: 信号配置
    
    Returns:
        signal: 交易信号，∈ [-1, +1]
        confidence: 置信度，∈ [0, 1]
        direction: 方向，∈ {-1, 0, +1}
    """
    long_score, short_score = action[0], action[1]
    
    # 计算自适应阈值
    max_score = max(long_score, short_score)
    threshold = config.base_threshold - config.threshold_k * max_score
    threshold = max(config.threshold_min, min(config.threshold_max, threshold))
    
    min_confidence = config.min_confidence
    if long_score > threshold and long_score > short_score:
        direction = +1
        confidence = long_score
    elif short_score > threshold and short_score > long_score:
        direction = -1
        confidence = short_score
    else:
        direction = 0
        confidence = 0.0
        
    # 成本硬约束
    if confidence > 0 and confidence < min_confidence:
        direction = 0
        confidence = 0.0
    
    signal = direction * confidence
    return signal, confidence, direction

def signal_position_risk(action: np.ndarray, config: Config, 
                                    net_position: float, 
                                    max_position: float) -> Tuple[float, float, int]:
    """
    基于净持仓风险的自适应阈值
    
    Args:
        action: [long_score, short_score], shape=(2,)
        config: 信号配置
        net_position: 当前净持仓
        max_position: 最大允许持仓
    
    Returns:
        signal: 交易信号，∈ [-1, +1]
        confidence: 置信度，∈ [0, 1]
        direction: 方向，∈ {-1, 0, +1}
    """
    long_score, short_score = action[0], action[1]
    # 计算自适应阈值（持仓多时提高阈值）
    position_ratio = abs(net_position) / max_position if max_position > 0 else 0.0
    threshold = config.base_threshold + config.threshold_k * position_ratio
    threshold = max(config.threshold_min, min(config.threshold_max, threshold))
    
    min_confidence = config.min_confidence
    
    # 判断方向和强度
    if long_score > threshold and long_score > short_score:
        direction = +1
        confidence = long_score
    elif short_score > threshold and short_score > long_score:
        direction = -1
        confidence = short_score
    else:
        direction = 0
        confidence = 0.0
    
    # 成本硬约束
    if confidence > 0 and confidence < min_confidence:
        direction = 0
        confidence = 0.0
    
    signal = direction * confidence
    return signal, confidence, direction



def to_signal(action: np.ndarray, config: Config, 
                     net_position: Optional[float] = None,
                     max_position: Optional[float] = None) -> Tuple[float, float, int]:
    """
    统一接口：将 SAC 动作转换为交易信号
    
    Args:
        action: [long_score, short_score], shape=(2,), 范围 [0, 1]
        config: 信号配置
        net_position: 当前净持仓（用于 position_risk 模式）
        max_position: 最大允许持仓（用于 position_risk 模式）
    
    Returns:
        signal: 交易信号，∈ [-1, +1]
        confidence: 置信度，∈ [0, 1]
        direction: 方向，∈ {-1, 0, +1}
    """
    # 确保 action 在 [0, 1] 范围内
    action = np.clip(action, 0.0, 1.0)
    
    if config.threshold_mode == 'fixed':
        return signal_fixed(action, config)
    elif config.threshold_mode == 'score_diff':
        return signal_score_diff(action, config)
    elif config.threshold_mode == 'score_strength':
        return signal_score_strength(action, config)
    elif config.threshold_mode == 'position_risk':
        if net_position is None or max_position is None:
            raise ValueError("position_risk mode requires net_position and max_position")
        return signal_position_risk(action, config, net_position, max_position)
    else:
        raise ValueError(f"Unknown threshold_mode: {config.threshold_mode}")
    
    
def calculate_cost(signal: float, confidence: float, config: Config) -> float:
    """
    计算交易成本
    
    Args:
        signal: 交易信号
        confidence: 置信度
        config: 信号配置
    
    Returns:
        cost: 单边成本
    """
    if config.cost_mode == 'fixed':
        return config.base_cost
    elif config.cost_mode == 'proportional':
        return config.base_cost * abs(signal) if signal != 0 else 0.0
    else:
        raise ValueError(f"Unknown cost_mode: {config.cost_mode}")
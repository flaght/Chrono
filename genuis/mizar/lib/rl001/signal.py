import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class Config:
    # 阈值模式
    threshold_mode: str  # 'fixed', 'score_diff', 'score_strength', 'position_risk'

    temperature: float      # Softmax 温度
    cash_score: float       # 观望锚点
    
    
    # ===== 阈值参数 =====
    threshold: float        # 基准阈值（所有模式都用）
    threshold_k: float      # 自适应系数（score_diff/score_strength/position_risk）
    threshold_min: float    # 自适应阈值下限
    threshold_max: float    # 自适应阈值上限
    
    # ===== 成本参数 =====
    base_cost: float
    cost_multiplier: float
    cost_mode: str
    # 分数转换模式：'conservative' 使用原始[0,1]分数；'aggressive' 使用logit映射
    score_mapping: str = 'conservative'
    
    
    def __post_init__(self):
        """计算 min_confidence"""
        if self.score_mapping not in ('conservative', 'aggressive'):
            raise ValueError(f"Unknown score_mapping: {self.score_mapping}")
        self.min_confidence = self.cost_multiplier * self.base_cost


def _build_scores(action: np.ndarray, cash_score: float, score_mapping: str) -> np.ndarray:
    """
    构建 softmax 输入分数。

    - conservative: 使用原始[0,1]分数（含clip，兼容旧逻辑）
    - aggressive: 先映射到 logit 空间，恢复动态范围
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != 2:
        raise ValueError(f"Action must have shape (2,), got {action.shape}")

    if score_mapping == 'conservative':
        a = np.clip(action, 0.0, 1.0)
        return np.array([float(a[0]), float(a[1]), float(cash_score)], dtype=np.float32)
    if score_mapping == 'aggressive':
        eps = 1e-4
        a = np.clip(action, eps, 1.0 - eps)
        long_score_raw, short_score_raw = np.log(a / (1.0 - a))
        if 0.0 <= cash_score <= 1.0:
            c = float(np.clip(cash_score, eps, 1.0 - eps))
            cash = float(np.log(c / (1.0 - c)))
        else:
            cash = float(cash_score)
        return np.array([long_score_raw, short_score_raw, cash], dtype=np.float32)

    raise ValueError(f"Unknown score_mapping: {score_mapping}")
        
        
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
    
    scores = _build_scores(action, config.cash_score, config.score_mapping)
    exp_scores = np.exp((scores - np.max(scores)) / config.temperature)
    softmax_scores = exp_scores / np.sum(exp_scores)
    
    long_prob = softmax_scores[0]
    short_prob = softmax_scores[1]
    
    threshold = config.threshold
    min_confidence = config.min_confidence
    
    # 按照相互排斥的 Softmax 分配后的真实可信度来判断
    if long_prob > threshold and long_prob > short_prob:
        direction = +1
        confidence = long_prob
    elif short_prob > threshold and short_prob > long_prob:
        direction = -1
        confidence = short_prob
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
    """
    scores = _build_scores(action, config.cash_score, config.score_mapping)
    exp_scores = np.exp((scores - np.max(scores)) / config.temperature)
    softmax_scores = exp_scores / np.sum(exp_scores)
    
    long_prob, short_prob = softmax_scores[0], softmax_scores[1]
    
    # 计算自适应阈值
    diff = abs(long_prob - short_prob)
    threshold = config.threshold - config.threshold_k * diff
    threshold = max(config.threshold_min, min(config.threshold_max, threshold))

    min_confidence = config.min_confidence
    
    # 判断方向和强度
    if long_prob > threshold and long_prob > short_prob:
        direction = +1
        confidence = long_prob
    elif short_prob > threshold and short_prob > long_prob:
        direction = -1
        confidence = short_prob
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
    """
    scores = _build_scores(action, config.cash_score, config.score_mapping)
    exp_scores = np.exp((scores - np.max(scores)) / config.temperature)
    softmax_scores = exp_scores / np.sum(exp_scores)
    
    long_prob, short_prob = softmax_scores[0], softmax_scores[1]
    
    # 计算自适应阈值
    max_score = max(long_prob, short_prob)
    threshold = config.threshold - config.threshold_k * max_score
    threshold = max(config.threshold_min, min(config.threshold_max, threshold))
    
    min_confidence = config.min_confidence
    if long_prob > threshold and long_prob > short_prob:
        direction = +1
        confidence = long_prob
    elif short_prob > threshold and short_prob > long_prob:
        direction = -1
        confidence = short_prob
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
    """
    # Softmax 互相排斥
    scores = _build_scores(action, config.cash_score, config.score_mapping)
    exp_scores = np.exp((scores - np.max(scores)) / config.temperature)
    softmax_scores = exp_scores / np.sum(exp_scores)
    
    long_prob, short_prob = softmax_scores[0], softmax_scores[1]
    
    # 计算自适应阈值（持仓多时提高阈值）
    position_ratio = abs(net_position) / max_position if max_position > 0 else 0.0
    threshold = config.threshold + config.threshold_k * position_ratio
    threshold = max(config.threshold_min, min(config.threshold_max, threshold))
    
    min_confidence = config.min_confidence
    
    # 判断方向和强度
    if long_prob > threshold and long_prob > short_prob:
        direction = +1
        confidence = long_prob
    elif short_prob > threshold and short_prob > long_prob:
        direction = -1
        confidence = short_prob
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
        action: [long_score, short_score], shape=(2,)
        config: 信号配置
        net_position: 当前净持仓（用于 position_risk 模式）
        max_position: 最大允许持仓（用于 position_risk 模式）
    
    Returns:
        signal: 交易信号，∈ [-1, +1]
        confidence: 置信度，∈ [0, 1]
        direction: 方向，∈ {-1, 0, +1}
    """
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

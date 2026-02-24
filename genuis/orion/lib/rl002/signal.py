import numpy as np
from typing import Tuple
from dataclasses import dataclass

@dataclass
class Config:
    # 权重参数
    min_weight: float = 0.0       # 最小权重 (A股: 0, 不能做空)
    max_weight: float = 1.0       # 单只股票最大权重上限
    normalize: bool = True        # 是否归一化权重 (权重之和 = 1)
    top_k: int = 0                # 只选前 k 只 (0 = 不限制, 全部分配)
    
    # 成本参数
    cost_rate: float = 0.0003     # 单边手续费率 (A股约万三)
    stamp_duty: float = 0.0005    # 印花税 (卖出时收取, 千分之0.5)
    
    
    # 奖励参数
    turnover_penalty: float = 0.0  # 额外的换手惩罚系数 (可选)
    
    # 调仓频率
    rebalance_window: int = 1     # 调仓间隔 (1 = 每步都可调仓)
    
def action_to_weights(action: np.ndarray, config: Config) -> np.ndarray:
    """
    将 SAC 动作转换为组合权重
    
    Args:
        action: shape=(N,), 每只股票的原始分数, 范围 [0, 1]
        config: 配置
    
    Returns:
        weights: shape=(N,), 归一化后的权重, 范围 [0, 1], 和为 1
    """
    
    # 确保非负
    weights = np.clip(action, config.min_weight, config.max_weight)
    
    # top_k 筛选: 只保留分数最高的 k 只
    if config.top_k > 0 and config.top_k < len(weights):
        top_indices = np.argsort(weights)[-config.top_k:]
        mask = np.zeros_like(weights)
        mask[top_indices] = 1.0
        weights = weights * mask
        
    # 归一化: 权重之和 = 1
    if config.normalize:
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            # 所有权重为 0, 均匀分配 (或保持全空仓)
            weights = np.zeros_like(weights)
    
    return weights


def calculate_turnover(old_weights: np.ndarray, new_weights: np.ndarray) -> float:
    """
    计算换手率
    
    Args:
        old_weights: 上一期权重, shape=(N,)
        new_weights: 新一期权重, shape=(N,)
    
    0 表示不调仓, 1 表示完全换仓
    """
    return np.sum(np.abs(new_weights - old_weights)) / 2.0


def calculate_transaction_cost(old_weights: np.ndarray, 
                                new_weights: np.ndarray, 
                                config: Config) -> float:
    """
    计算调仓的交易成本
    
    A股成本:
      - 买入: cost_rate (佣金)
      - 卖出: cost_rate (佣金) + stamp_duty (印花税)
    
    total_cost: 总交易成本 (占组合的比例)
    """
    
    weight_changes = new_weights - old_weights
    
    # 买入部分 (权重增加): 支付佣金
    buy_amount = np.sum(np.maximum(weight_changes, 0))
    buy_cost = buy_amount * config.cost_rate
    
    # 卖出部分 (权重减少): 支付佣金 + 印花税
    sell_amount = np.sum(np.abs(np.minimum(weight_changes, 0)))
    sell_cost = sell_amount * (config.cost_rate + config.stamp_duty)
    
    return buy_cost + sell_cost


def calculate_portfolio_return(weights: np.ndarray, returns: np.ndarray) -> float:
    """
    计算组合收益
        weights: 当期权重, shape=(N,)
        returns: 当期各股票收益率, shape=(N,)
    
    portfolio_return: 组合收益率 = sum(w_i * r_i)
    """
    return np.dot(weights, returns)

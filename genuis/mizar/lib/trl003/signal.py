"""
数字货币期现正套信号转换

核心设计:
  - 正套: 做多现货 + 做空期货 (方向固定, 不存在反套)
  - 连续权重: agent 为每个交易对输出 [0, 1] 的仓位权重
  - 收益定义 (log return):
      y_basis = log(S₂/S₁) - log(F₂/F₁)    ← 基差对数收益
      R_total = (exp(y_basis) - 1) + f - c   ← simple 口径合成
      y_total = log(1 + R_total)             ← 转回 log return
  - 正套盈利时 y_basis > 0 (现货跑赢期货)，无需取负
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class Config:
    """期现套利配置"""
    
    # 权重参数
    max_weight: float = 1.0       # 单个交易对最大权重
    normalize: bool = True        # 是否归一化 (权重之和 = 1)
    top_k: int = 0                # 只选前 k 个交易对 (0 = 不限制)
    
    # 成本参数
    spot_fee: float = 0.0001      # 现货手续费 (币安 VIP: 万一)
    futures_fee: float = 0.0002   # 期货手续费 (taker 约万二)
    
    # 开仓成本 = spot_fee + futures_fee (双边)
    # 平仓成本 = spot_fee + futures_fee (双边)
    
    # 基差阈值
    min_basis_pct: float = 0.001  # 最小开仓基差 (0.1%), 基差太小不值得做
    
    # 换手惩罚
    turnover_penalty: float = 0.0
    
def action_to_weights(action: np.ndarray, config: Config) -> np.ndarray:
    """
    将 SAC 动作转换为套利仓位权重
    
    Args:
        action: shape=(N,), 每个交易对的原始分数, 范围 [0, 1]
        config: 配置
    
    Returns:
        weights: shape=(N,), 归一化后的权重, 和为 1
    """
    # 非负 (正套权重只有 [0, 1])
    weights = np.clip(action, 0.0, config.max_weight)
    
    # top_k 筛选
    if config.top_k > 0 and config.top_k < len(weights):
        top_indices = np.argsort(weights)[-config.top_k:]
        mask = np.zeros_like(weights)
        mask[top_indices] = 1.0
        weights = weights * mask
    
    # 归一化
    if config.normalize:
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            weights = np.zeros_like(weights)
    
    return weights

def calculate_turnover(old_weights: np.ndarray, new_weights: np.ndarray) -> float:
    """
    计算换手率
    
    Returns:
        turnover: sum(|w_new - w_old|) / 2, 范围 [0, 1]
    """
    return np.sum(np.abs(new_weights - old_weights)) / 2.0


def calculate_transaction_cost(old_weights: np.ndarray,
                                new_weights: np.ndarray,
                                config: Config) -> float:
    """
    计算调仓的交易成本
    
    正套开仓: 买现货 (spot_fee) + 卖期货 (futures_fee)
    正套平仓: 卖现货 (spot_fee) + 买期货 (futures_fee)
    
    两边都要收费, 所以单次调仓成本 = (spot_fee + futures_fee) * 调仓量
    """
    weight_changes = np.abs(new_weights - old_weights)
    total_change = np.sum(weight_changes)
    
    # 双边成本 (现货 + 期货)
    cost_per_unit = config.spot_fee + config.futures_fee
    return total_change * cost_per_unit

def calculate_arbitrage_return(weights: np.ndarray, 
                               log_returns: np.ndarray) -> float:
    """
    计算正套组合收益
    
    收益定义:
        log_return = log(S₂/S₁) - log(F₂/F₁)
        正值 = 正套盈利 (现货跑赢期货, 基差收敛)
        负值 = 正套亏损
    
    注意: 输入已经是正确符号的 log return, 无需取负
    
    Args:
        weights: 当期权重, shape=(N,)
        log_returns: 当期各交易对的正套 log return, shape=(N,)
    
    Returns:
        portfolio_return: 加权组合收益
    """
    return np.dot(weights, log_returns)
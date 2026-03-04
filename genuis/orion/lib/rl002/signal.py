from dataclasses import dataclass
import numpy as np

@dataclass
class Config:
    # 权重参数
    min_weight: float = 0.0       # 最小权重 (A股: 0, 不能做空)
    max_weight: float = 1.0       # 单只股票最大权重上限
    normalize: bool = True        # 是否归一化权重 (可行约束下权重之和 = 1)
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
    将 SAC 动作转换为组合权重 (具备 Softmax 平滑分配，防梯度死亡)
    
    Args:
        action: shape=(N,), 每只股票的原始分数, 范围 [0, 1]
        config: 配置
    
    Returns:
        weights: shape=(N,), 归一化后的权重, 范围 [0, 1]
    """
    raw_scores = np.asarray(action, dtype=np.float32).flatten()
    if raw_scores.size == 0:
        return raw_scores

    # Use unclipped action ranking for top-k selection to avoid tie collapse caused by max_weight clipping.
    rank_scores = np.clip(raw_scores, config.min_weight, 1.0)
    if config.top_k > 0 and config.top_k < len(rank_scores):
        effective_k = int(config.top_k)
    else:
        effective_k = int(len(rank_scores))

    if config.normalize and config.max_weight > 0 and effective_k * config.max_weight < 1.0 - 1e-12:
        raise ValueError(
            f"Infeasible allocation: top_k({effective_k}) * max_weight({config.max_weight}) < 1.0"
        )

    if config.top_k > 0 and config.top_k < len(rank_scores):
        top_indices = np.argsort(rank_scores)[-config.top_k:]
        sparse_scores = np.zeros_like(rank_scores)
        sparse_scores[top_indices] = rank_scores[top_indices]
        weights = sparse_scores
        candidate_mask = np.zeros_like(rank_scores, dtype=bool)
        candidate_mask[top_indices] = True
    else:
        weights = rank_scores.copy()
        candidate_mask = np.ones_like(rank_scores, dtype=bool)

    if config.normalize:
        active_mask = weights > 0
        if not np.any(active_mask):
            active_mask = candidate_mask

        if np.any(active_mask):
            # Softmax on active names keeps ranking information and avoids hard mask brittleness.
            temperature = 0.05
            logits = weights[active_mask]
            logits = logits - np.max(logits)
            exp_scores = np.exp(logits / temperature)
            normalized = exp_scores / np.sum(exp_scores)
            weights = np.zeros_like(weights)
            weights[active_mask] = normalized
        else:
            return np.zeros_like(weights)

        # Enforce per-asset max weight after normalization.
        if config.max_weight > 0:
            cap = float(config.max_weight)
            free = np.ones_like(weights, dtype=bool)
            projected = np.zeros_like(weights)
            remaining_mass = 1.0

            while remaining_mass > 1e-12 and np.any(free):
                current = weights[free]
                current_sum = current.sum()
                if current_sum <= 0:
                    break
                scaled = current / current_sum * remaining_mass
                over_mask = scaled > cap + 1e-12
                free_indices = np.where(free)[0]
                if not np.any(over_mask):
                    projected[free_indices] = scaled
                    remaining_mass = 0.0
                    break

                over_indices = free_indices[over_mask]
                projected[over_indices] = cap
                remaining_mass -= cap * len(over_indices)
                free[over_indices] = False

            weights = np.clip(projected, 0.0, cap)
    else:
        weights = np.clip(weights, config.min_weight, config.max_weight)

    return weights

def calculate_turnover(old_weights: np.ndarray, new_weights: np.ndarray) -> float:
    return np.sum(np.abs(new_weights - old_weights)) / 2.0

def calculate_transaction_cost(old_weights: np.ndarray, new_weights: np.ndarray, config: Config) -> float:
    weight_changes = new_weights - old_weights
    buy_amount = np.sum(np.maximum(weight_changes, 0))
    buy_cost = buy_amount * config.cost_rate
    sell_amount = np.sum(np.abs(np.minimum(weight_changes, 0)))
    sell_cost = sell_amount * (config.cost_rate + config.stamp_duty)
    return buy_cost + sell_cost

def calculate_portfolio_return(weights: np.ndarray, returns: np.ndarray) -> float:
    return np.dot(weights, returns)

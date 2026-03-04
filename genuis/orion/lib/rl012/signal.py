from dataclasses import dataclass
import pdb
import numpy as np


@dataclass
class Config:
    # 权重约束
    min_weight: float           # 最小权重 (A股: 0, 不能做空)
    max_weight: float           # 单只股票最大权重上限
    normalize: bool             # 是否归一化权重 (权重之和 = 1)
    top_k: int                  # 只选前 k 只 (0 = 不限制)

    # 交易成本
    cost_rate: float            # 交易佣金率
    stamp_duty: float           # 印花税率

    # 惩罚 / 调仓
    turnover_penalty: float     # 换手惩罚系数
    rebalance_window: int       # 调仓窗口 (1 = 每步调仓)

    # Softmax
    softmax_temperature: float  # scores_to_weights 内 softmax 温度


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average rank for ties, 0-based ranks."""
    x = np.asarray(values, dtype=np.float64).flatten()
    n = x.size
    if n == 0:
        return x.astype(np.float32)

    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks.astype(np.float32)

def rank_ic(scores: np.ndarray, returns: np.ndarray) -> float:
    """Spearman-style rank IC."""
    s = np.asarray(scores, dtype=np.float64).flatten()
    r = np.asarray(returns, dtype=np.float64).flatten()
    n = min(s.size, r.size)
    if n < 2:
        return 0.0
    s = s[:n]
    r = r[:n]

    rs = _rankdata(s).astype(np.float64)
    rr = _rankdata(r).astype(np.float64)

    rs -= rs.mean()
    rr -= rr.mean()
    denom = np.sqrt(np.sum(rs * rs) * np.sum(rr * rr))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(rs * rr) / denom)

def scores_to_weights(
    scores: np.ndarray,
    config: Config,
    top_k: int = 0,
) -> np.ndarray:
    """
    Convert raw scores to long-only portfolio weights.
    """
    raw_scores = np.asarray(scores, dtype=np.float32).flatten()
    if raw_scores.size == 0:
        return raw_scores

    rank_scores = np.clip(raw_scores, config.min_weight, 1.0)

    effective_top_k = top_k if top_k > 0 else config.top_k
    if effective_top_k > 0 and effective_top_k < rank_scores.size:
        candidate_indices = np.argsort(rank_scores)[-effective_top_k:]
    else:
        candidate_indices = np.arange(rank_scores.size, dtype=np.int64)

    if (
        config.normalize
        and config.max_weight > 0
        and candidate_indices.size * config.max_weight < 1.0 - 1e-12
    ):
        raise ValueError(
            f"Infeasible allocation: top_k({candidate_indices.size}) * "
            f"max_weight({config.max_weight}) < 1.0"
        )

    weights = np.zeros_like(rank_scores)
    candidate_scores = rank_scores[candidate_indices]

    # Softmax keeps ranking information
    logits = candidate_scores - np.max(candidate_scores)
    exp_scores = np.exp(logits / config.softmax_temperature)
    soft = exp_scores / np.sum(exp_scores)
    weights[candidate_indices] = soft.astype(np.float32)

    if config.max_weight > 0:
        cap = float(config.max_weight)
        free = np.ones_like(weights, dtype=bool)
        projected = np.zeros_like(weights)
        remaining_mass = 1.0

        while remaining_mass > 1e-12 and np.any(free):
            current = weights[free]
            current_sum = current.sum()
            if current_sum <= 1e-12:
                break
            scaled = current / current_sum * remaining_mass
            over = scaled > cap + 1e-12
            free_indices = np.where(free)[0]
            if not np.any(over):
                projected[free_indices] = scaled
                remaining_mass = 0.0
                break
            over_indices = free_indices[over]
            projected[over_indices] = cap
            remaining_mass -= cap * over_indices.size
            free[over_indices] = False
        weights = np.clip(projected, 0.0, cap)

    if not config.normalize:
        weights = np.clip(weights, config.min_weight, config.max_weight)

    return weights.astype(np.float32)

def calculate_turnover(old_weights: np.ndarray, new_weights: np.ndarray) -> float:
    return float(np.sum(np.abs(new_weights - old_weights)) / 2.0)


def calculate_transaction_cost(
    old_weights: np.ndarray, new_weights: np.ndarray, config: Config
) -> float:
    weight_changes = new_weights - old_weights
    buy_amount = np.sum(np.maximum(weight_changes, 0.0))
    sell_amount = np.sum(np.abs(np.minimum(weight_changes, 0.0)))
    buy_cost = buy_amount * config.cost_rate
    sell_cost = sell_amount * (config.cost_rate + config.stamp_duty)
    return float(buy_cost + sell_cost)

def calculate_portfolio_return(weights: np.ndarray, returns: np.ndarray) -> float:
    return float(np.dot(weights, returns))

"""根据 Optuna trial 在指定特征和算子空间内采样公式。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from miner.formula import (
    Formula,
    MiningMode,
    validate_names,
)
from miner.operators import (
    DEFAULT_BINARY_OPERATORS,
    DEFAULT_PAIR_OPERATORS,
    DEFAULT_UNARY_OPERATORS,
    DEFAULT_WINDOW_OPERATORS,
)

def sample_formula(
    trial: Any,
    *,
    feature_names: Sequence[str],
    mode: MiningMode = "free",
    periods: Sequence[int] = (5, 10, 20, 40, 60),
    max_depth: int = 3,
    unary_operators: Sequence[str] = DEFAULT_UNARY_OPERATORS,
    window_operators: Sequence[str] = DEFAULT_WINDOW_OPERATORS,
    binary_operators: Sequence[str] = DEFAULT_BINARY_OPERATORS,
    pair_operators: Sequence[str] = DEFAULT_PAIR_OPERATORS,
) -> Formula:
    """从兼容 Optuna 的 trial 中采样一个确定性公式。"""
    features = validate_names(feature_names, "feature_names")
    if mode not in ("free", "directed"):
        raise ValueError("mode 必须为 'free' 或 'directed'")
    if max_depth < 1:
        raise ValueError("max_depth 必须大于等于 1")
    normalized_periods = tuple(dict.fromkeys(periods))
    if not normalized_periods or any(
        not isinstance(period, int) or isinstance(period, bool) or period <= 0
        for period in normalized_periods
    ):
        raise ValueError("periods 必须为正整数集合")

    all_kinds: list[str] = []
    if unary_operators:
        all_kinds.append("unary")
    if window_operators:
        all_kinds.append("window")
    if binary_operators:
        all_kinds.append("binary")
    if pair_operators:
        all_kinds.append("pair")
    if not all_kinds:
        raise ValueError("至少需要指定一个算子")

    def choose_leaf(name: str) -> Formula:
        # feature_names 在两种模式下都是完整且唯一的特征搜索空间。
        return Formula.leaf(trial.suggest_categorical(name, list(features)))

    # 定向与自由模式共用公式构造过程，差异由上层传入的算子池决定。
    root = choose_leaf("anchor")
    depth = trial.suggest_int("depth", 1, max_depth)
    for level in range(depth):
        kind = trial.suggest_categorical(f"kind_{level}", all_kinds)
        if kind == "unary":
            operator = trial.suggest_categorical(
                f"unary_operator_{level}", list(unary_operators))
            root = Formula(operator, (root,))
        elif kind == "window":
            operator = trial.suggest_categorical(
                f"window_operator_{level}", list(window_operators))
            period = trial.suggest_categorical(
                f"period_{level}", list(normalized_periods))
            root = Formula(operator, (root,), period=period)
        elif kind == "binary":
            operator = trial.suggest_categorical(
                f"binary_operator_{level}", list(binary_operators))
            right = choose_leaf(f"right_{level}")
            root = Formula(operator, (root, right))
        else:
            operator = trial.suggest_categorical(
                f"pair_operator_{level}", list(pair_operators))
            right = choose_leaf(f"right_{level}")
            period = trial.suggest_categorical(
                f"period_{level}", list(normalized_periods))
            root = Formula(operator, (root, right), period=period)
    return root

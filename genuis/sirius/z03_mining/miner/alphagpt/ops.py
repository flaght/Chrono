"""把公共算子池展开为 AlphaGPT 模型使用的 Token 注册表。"""

from __future__ import annotations

from dataclasses import dataclass

from miner.operators import OperatorPool


@dataclass(frozen=True)
class OperatorSpec:
    token_name: str
    operator: str
    arity: int
    period: int | None = None


def build_operator_specs(
    periods: tuple[int, ...],
    operator_pool: OperatorPool,
) -> tuple[OperatorSpec, ...]:
    """把公共算子池按周期展开为 AlphaGPT 模型 Token。"""
    candidates = [
        *(OperatorSpec(name, name, 1) for name in operator_pool.unary),
        *(OperatorSpec(name, name, 2) for name in operator_pool.binary),
    ]
    for period in periods:
        candidates.extend(
            OperatorSpec(f"{name}_{period}", name, 1, period)
            for name in operator_pool.window
        )
        candidates.extend(
            OperatorSpec(f"{name}_{period}", name, 2, period)
            for name in operator_pool.pair
        )
    return tuple(candidates)


__all__ = [
    "OperatorSpec", "build_operator_specs",
]

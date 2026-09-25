"""不依赖行情或撮合框架的三腿价差计算。"""

from __future__ import annotations

from math import log, sqrt
from typing import Mapping


def normalized_log_spread(prices: Mapping[str, float], baseline: Mapping[str, float],
                          anchor: str, hedges: tuple[str, str],
                          weights: tuple[float, float]) -> float:
    spread = log(prices[anchor] / baseline[anchor])
    for product, weight in zip(hedges, weights):
        spread -= weight * log(prices[product] / baseline[product])
    return spread


def score(current: float, previous: tuple[float, ...]) -> float:
    mean = sum(previous) / len(previous)
    variance = sum((value - mean) ** 2 for value in previous) / len(previous)
    return (current - mean) / sqrt(variance) if variance > 0 else 0.0


def direction_for(z: float, previous_direction: int, entry_z: float,
                  exit_z: float) -> int:
    if abs(z) <= exit_z:
        return 0
    if z <= -entry_z:
        return 1
    if z >= entry_z:
        return -1
    return previous_direction

"""进化算法因子挖掘器配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ScoreName = Literal["abs_ic_mean", "ic_sharpe"]


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 200
    generations: int = 30
    tournament_size: int = 5
    elite_size: int = 10
    max_depth: int = 3
    periods: tuple[int, ...] = (5, 10, 20, 40, 60)
    crossover_rate: float = 0.70
    subtree_mutation_rate: float = 0.15
    point_mutation_rate: float = 0.10
    hoist_mutation_rate: float = 0.05
    min_observations: int = 100
    score: ScoreName = "abs_ic_mean"
    seed: int = 42
    log_interval: int = 1
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size 必须大于等于 2")
        if self.generations < 1:
            raise ValueError("generations 必须为正整数")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size 必须位于 [2, population_size]")
        if not 1 <= self.elite_size < self.population_size:
            raise ValueError("elite_size 必须位于 [1, population_size)")
        if self.max_depth < 1:
            raise ValueError("max_depth 必须大于等于 1")
        if self.min_observations < 1:
            raise ValueError("min_observations 必须为正整数")
        if self.log_interval < 1:
            raise ValueError("log_interval 必须为正整数")
        if self.score not in ("abs_ic_mean", "ic_sharpe"):
            raise ValueError("score 必须为 'abs_ic_mean' 或 'ic_sharpe'")
        if not self.periods or any(
            not isinstance(period, int) or isinstance(period, bool) or period <= 0
            for period in self.periods
        ):
            raise ValueError("periods 必须为正整数集合")
        rates = (
            self.crossover_rate,
            self.subtree_mutation_rate,
            self.point_mutation_rate,
            self.hoist_mutation_rate,
        )
        if any(rate < 0 or rate > 1 for rate in rates):
            raise ValueError("交叉和变异概率必须位于 [0, 1]")
        if sum(rates) > 1.0 + 1e-12:
            raise ValueError("交叉和变异概率之和不能大于 1")


__all__ = ["EvolutionConfig", "ScoreName"]

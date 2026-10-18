"""贝叶斯因子挖掘器配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ScoreName = Literal["abs_ic_mean", "ic_sharpe"]


@dataclass(frozen=True)
class BayesianConfig:
    periods: tuple[int, ...] = (5, 10, 20, 40, 60)
    max_depth: int = 3
    min_observations: int = 100
    score: ScoreName = "abs_ic_mean"
    sampler_seed: int = 42
    n_trials: int = 1000
    n_jobs: int = 4
    log_interval: int = 50
    verbose: bool = True

    def __post_init__(self) -> None:
        if not self.periods or any(
            not isinstance(period, int) or isinstance(period, bool) or period <= 0
            for period in self.periods
        ):
            raise ValueError("periods 必须为正整数集合")
        if self.max_depth < 1:
            raise ValueError("max_depth 必须大于等于 1")
        if self.min_observations < 1:
            raise ValueError("min_observations 必须为正整数")
        if self.score not in ("abs_ic_mean", "ic_sharpe"):
            raise ValueError("score 必须为 'abs_ic_mean' 或 'ic_sharpe'")
        if self.n_trials < 1:
            raise ValueError("n_trials 必须为正整数")
        if self.n_jobs == 0:
            raise ValueError("n_jobs 不能为 0")
        if self.log_interval < 1:
            raise ValueError("log_interval 必须为正整数")


__all__ = ["BayesianConfig", "ScoreName"]

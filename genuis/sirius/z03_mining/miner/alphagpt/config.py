"""AlphaGPT 因子挖掘器配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ScoreName = Literal["abs_ic_mean", "ic_sharpe"]


@dataclass(frozen=True)
class AlphaGPTConfig:
    batch_size: int = 64
    train_steps: int = 100
    max_formula_len: int = 4
    learning_rate: float = 1e-3
    periods: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
    min_observations: int = 100
    score: ScoreName = "abs_ic_mean"
    invalid_reward: float = 0.0
    seed: int = 42
    device: str | None = None
    log_interval: int = 1
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.train_steps < 1:
            raise ValueError("batch_size 和 train_steps 必须为正整数")
        if self.max_formula_len < 2:
            raise ValueError("max_formula_len 必须大于等于 2")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")
        if self.min_observations < 1:
            raise ValueError("min_observations 必须为正整数")
        if self.score not in ("abs_ic_mean", "ic_sharpe"):
            raise ValueError("score 必须为 'abs_ic_mean' 或 'ic_sharpe'")
        if self.log_interval < 1:
            raise ValueError("log_interval 必须为正整数")
        if not self.periods or any(
            not isinstance(period, int) or isinstance(period, bool) or period <= 0
            for period in self.periods
        ):
            raise ValueError("periods 必须为正整数集合")


__all__ = ["AlphaGPTConfig", "ScoreName"]

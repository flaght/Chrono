"""进化种群训练、公式评分与进度日志。"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from typing import Any

import polars as pl

from miner.evaluate import (
    EvaluationType,
    TimeSeriesEvaluationConfig,
    detect_evaluation_type,
    ensure_evaluation_supported,
    evaluate_formula,
)
from miner.formula import Formula
from miner.operators import OperatorPool

from .config import EvolutionConfig
from .genetic import FormulaGenetics


class EvolutionEngine:
    """通过精英保留和锦标赛选择迭代公式种群。"""

    def __init__(
        self,
        *,
        df_lazy: pl.LazyFrame,
        feature_names: Sequence[str],
        return_column: str,
        config: EvolutionConfig,
        operators: OperatorPool,
        evaluation_type: EvaluationType | None = None,
        evaluation_config: TimeSeriesEvaluationConfig | None = None,
    ) -> None:
        self.df_lazy = df_lazy
        self.return_column = return_column
        self.config = config
        self.evaluation_type = evaluation_type or detect_evaluation_type(df_lazy)
        ensure_evaluation_supported(self.evaluation_type)
        self.evaluation_config = evaluation_config or TimeSeriesEvaluationConfig()
        self.random = random.Random(config.seed)
        self.genetics = FormulaGenetics(
            feature_names=feature_names,
            periods=config.periods,
            max_depth=config.max_depth,
            operators=operators,
            random_state=self.random,
        )
        self.metrics_cache: dict[str, dict[str, float | int]] = {}
        self.records: dict[str, dict[str, Any]] = {}
        self.evaluation_errors = 0

    def train(self) -> list[dict[str, Any]]:
        started_at = time.perf_counter()
        population = self._initial_population()
        if self.config.verbose:
            print(
                "[Evolution] 开始进化："
                f"种群={self.config.population_size}，"
                f"代数={self.config.generations}，"
                f"锦标赛={self.config.tournament_size}，"
                f"精英={self.config.elite_size}",
                flush=True,
            )

        for generation in range(self.config.generations):
            scored = self._evaluate_population(population, generation)
            scored.sort(key=lambda item: item[0], reverse=True)
            completed = generation + 1
            if self._should_log(completed):
                elapsed = time.perf_counter() - started_at
                speed = completed / elapsed if elapsed > 0 else 0.0
                remaining = self.config.generations - completed
                eta = remaining / speed if speed > 0 else 0.0
                scores = [item[0] for item in scored]
                print(
                    f"[Evolution] 进度={completed}/{self.config.generations} "
                    f"最佳={scores[0]:.6f} "
                    f"平均={sum(scores) / len(scores):.6f} "
                    f"唯一候选={len(self.records)} "
                    f"实际评估={len(self.metrics_cache)} "
                    f"评估异常={self.evaluation_errors} "
                    f"耗时={elapsed:.1f}秒 预计剩余={eta:.1f}秒",
                    flush=True,
                )
            if completed < self.config.generations:
                population = self._next_population(scored)

        result = sorted(
            self.records.values(),
            key=lambda item: item["score"],
            reverse=True,
        )
        if self.config.verbose:
            print(
                f"[Evolution] 进化结束：累计有效候选={len(result)}",
                flush=True,
            )
        return result

    def _initial_population(self) -> list[Formula]:
        return [
            self.genetics.random_formula()
            for _ in range(self.config.population_size)
        ]

    def _evaluate_population(
        self,
        population: Sequence[Formula],
        generation: int,
    ) -> list[tuple[float, Formula]]:
        result = []
        log_interval = max(1, len(population) // 4)
        for index, formula in enumerate(population, start=1):
            metrics = self.metrics_cache.get(formula.factor_id)
            if metrics is None:
                try:
                    metrics = evaluate_formula(
                        self.df_lazy,
                        formula,
                        return_column=self.return_column,
                        min_observations=self.config.min_observations,
                        evaluation_type=self.evaluation_type,
                        time_series_config=self.evaluation_config,
                    )
                except (
                    pl.exceptions.PolarsError,
                    ValueError,
                    KeyError,
                    ZeroDivisionError,
                    OverflowError,
                ):
                    self.evaluation_errors += 1
                    metrics = {
                        "ic_mean": 0.0,
                        "abs_ic_mean": 0.0,
                        "ic_sharpe": 0.0,
                        "observations": 0,
                        "period_count": 0,
                    }
                self.metrics_cache[formula.factor_id] = metrics
            score = float(metrics[self.config.score])
            result.append((score, formula))
            previous = self.records.get(formula.factor_id)
            if previous is None or score > previous["score"]:
                self.records[formula.factor_id] = {
                    "generation": generation,
                    "score": score,
                    "formula_id": formula.factor_id,
                    "formula": formula.text,
                    "features": sorted(formula.features),
                    **metrics,
                }
            if self._should_log(generation + 1) and (
                index % log_interval == 0 or index == len(population)
            ):
                print(
                    f"[Evolution] 第{generation + 1}代评估="
                    f"{index}/{len(population)}",
                    flush=True,
                )
        return result

    def _next_population(
        self,
        scored: Sequence[tuple[float, Formula]],
    ) -> list[Formula]:
        next_population = [
            formula for _, formula in scored[:self.config.elite_size]
        ]
        while len(next_population) < self.config.population_size:
            roll = self.random.random()
            if roll < self.config.crossover_rate:
                child = self.genetics.crossover(
                    self._tournament(scored), self._tournament(scored)
                )
            elif roll < (
                self.config.crossover_rate
                + self.config.subtree_mutation_rate
            ):
                child = self.genetics.subtree_mutation(self._tournament(scored))
            elif roll < (
                self.config.crossover_rate
                + self.config.subtree_mutation_rate
                + self.config.point_mutation_rate
            ):
                child = self.genetics.point_mutation(self._tournament(scored))
            elif roll < (
                self.config.crossover_rate
                + self.config.subtree_mutation_rate
                + self.config.point_mutation_rate
                + self.config.hoist_mutation_rate
            ):
                child = self.genetics.hoist_mutation(self._tournament(scored))
            else:
                child = self._tournament(scored)
            next_population.append(child)
        return next_population

    def _tournament(
        self,
        scored: Sequence[tuple[float, Formula]],
    ) -> Formula:
        contestants = self.random.sample(
            list(scored), self.config.tournament_size
        )
        return max(contestants, key=lambda item: item[0])[1]

    def _should_log(self, completed_generation: int) -> bool:
        return self.config.verbose and (
            completed_generation == 1
            or completed_generation % self.config.log_interval == 0
            or completed_generation == self.config.generations
        )


__all__ = ["EvolutionEngine"]

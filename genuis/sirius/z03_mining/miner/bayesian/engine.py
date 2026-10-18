"""Optuna 试验执行、公式评分与进度日志。"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from miner.evaluate import (
    EvaluationType,
    TimeSeriesEvaluationConfig,
    detect_evaluation_type,
    ensure_evaluation_supported,
    evaluate_formula,
)
from miner.formula import MiningMode

from .config import BayesianConfig
from .sample import sample_formula


class BayesianEngine:
    """执行贝叶斯公式搜索并将试验结果转换为统一记录。"""

    def __init__(
        self,
        *,
        df_lazy: pl.LazyFrame,
        feature_names: Sequence[str],
        return_column: str,
        mode: MiningMode,
        config: BayesianConfig,
        operator_config: Mapping[str, Sequence[str]],
        evaluation_type: EvaluationType | None = None,
        evaluation_config: TimeSeriesEvaluationConfig | None = None,
    ) -> None:
        self.df_lazy = df_lazy
        self.feature_names = tuple(feature_names)
        self.return_column = return_column
        self.mode = mode
        self.config = config
        self.operator_config = dict(operator_config)
        self.evaluation_type = evaluation_type or detect_evaluation_type(df_lazy)
        ensure_evaluation_supported(self.evaluation_type)
        self.evaluation_config = evaluation_config or TimeSeriesEvaluationConfig()

    def train(
        self,
        *,
        study_name: str | None = None,
        storage: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            import optuna
        except ImportError as error:
            raise RuntimeError(
                "贝叶斯挖掘器需要 Optuna: pip install optuna"
            ) from error

        sampler = optuna.samplers.TPESampler(seed=self.config.sampler_seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=study_name,
            storage=storage,
            load_if_exists=storage is not None,
        )
        started_at = time.perf_counter()
        completed_trials = 0
        progress_lock = threading.Lock()

        if self.config.verbose:
            print(
                "[Bayesian] 开始寻优："
                f"试验={self.config.n_trials}，并发={self.config.n_jobs}，"
                f"最大深度={self.config.max_depth}，模式={self.mode}，"
                f"特征={self.feature_names}",
                flush=True,
            )

        def objective(trial: Any) -> float:
            formula = sample_formula(
                trial,
                feature_names=self.feature_names,
                mode=self.mode,
                periods=self.config.periods,
                max_depth=self.config.max_depth,
                **self.operator_config,
            )
            metrics = evaluate_formula(
                self.df_lazy,
                formula,
                return_column=self.return_column,
                min_observations=self.config.min_observations,
                evaluation_type=self.evaluation_type,
                time_series_config=self.evaluation_config,
            )
            trial.set_user_attr("formula", formula.text)
            trial.set_user_attr("formula_id", formula.factor_id)
            trial.set_user_attr("features", sorted(formula.features))
            for name, value in metrics.items():
                trial.set_user_attr(name, value)
            return float(metrics[self.config.score])

        def log_progress(current_study: Any, _: Any) -> None:
            nonlocal completed_trials
            if not self.config.verbose:
                return
            with progress_lock:
                completed_trials += 1
                should_log = (
                    completed_trials == 1
                    or completed_trials % self.config.log_interval == 0
                    or completed_trials == self.config.n_trials
                )
                if not should_log:
                    return
                elapsed = time.perf_counter() - started_at
                speed = completed_trials / elapsed if elapsed > 0 else 0.0
                remaining = self.config.n_trials - completed_trials
                eta = remaining / speed if speed > 0 else 0.0
                try:
                    best = float(current_study.best_value)
                except ValueError:
                    best = 0.0
                print(
                    f"[Bayesian] 进度={completed_trials}/"
                    f"{self.config.n_trials} 最佳={best:.6f} "
                    f"耗时={elapsed:.1f}秒 预计剩余={eta:.1f}秒",
                    flush=True,
                )

        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            callbacks=[log_progress],
            catch=(
                pl.exceptions.PolarsError,
                ValueError,
                KeyError,
                ZeroDivisionError,
                OverflowError,
            ),
        )

        records = []
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            attrs = trial.user_attrs
            record = {
                "trial": trial.number,
                "score": trial.value,
                "formula_id": attrs.get("formula_id"),
                "formula": attrs.get("formula"),
                "features": attrs.get("features"),
                "ic_mean": attrs.get("ic_mean"),
                "abs_ic_mean": attrs.get("abs_ic_mean"),
                "ic_sharpe": attrs.get("ic_sharpe"),
                "observations": attrs.get("observations"),
                "period_count": attrs.get("period_count"),
                "params": json.dumps(
                    trial.params, ensure_ascii=False, sort_keys=True
                ),
            }
            # 保留 cux001 返回的完整时序评估明细，便于后续筛选和诊断。
            for name, value in attrs.items():
                if name not in {"formula", "formula_id", "features"}:
                    record[name] = value
            records.append(record)
        if self.config.verbose:
            print(
                f"[Bayesian] 寻优结束：成功试验={len(records)}",
                flush=True,
            )
        return records


__all__ = ["BayesianEngine"]

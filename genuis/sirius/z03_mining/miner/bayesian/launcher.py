"""贝叶斯因子挖掘公开入口。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import polars as pl

from miner.evaluate import TimeSeriesEvaluationConfig, detect_evaluation_type, ensure_evaluation_supported
from miner.formula import (
    MiningMode,
    validate_names,
)
from miner.operators import build_operator_pool

from .config import BayesianConfig
from .engine import BayesianEngine


class Launcher:
    """提供与 AlphaGPT、Evolution 后端一致的 optimize 接口。"""

    def __init__(
        self,
        *,
        feature_names: Iterable[str],
        return_column: str,
        mode: MiningMode = "free",
        config: BayesianConfig | None = None,
        operator_config: Mapping[str, Sequence[str]] | None = None,
        periods: Iterable[int] | None = None,
        max_depth: int | None = None,
        min_observations: int | None = None,
        score: Literal["abs_ic_mean", "ic_sharpe"] | None = None,
        sampler_seed: int | None = None,
        evaluation_config: TimeSeriesEvaluationConfig | None = None,
    ) -> None:
        self.feature_names = validate_names(feature_names, "feature_names")
        self.return_column = return_column
        self.mode = mode
        self.evaluation_config = evaluation_config or TimeSeriesEvaluationConfig()

        values = dict((config or BayesianConfig()).__dict__)
        legacy_overrides = {
            "periods": tuple(dict.fromkeys(periods)) if periods is not None else None,
            "max_depth": max_depth,
            "min_observations": min_observations,
            "score": score,
            "sampler_seed": sampler_seed,
        }
        values.update({
            name: value for name, value in legacy_overrides.items()
            if value is not None
        })
        self.config = BayesianConfig(**values)
        if mode not in ("free", "directed"):
            raise ValueError("mode 必须为 'free' 或 'directed'")
        if mode == "directed" and operator_config is None:
            raise ValueError(
                "定向挖掘必须显式指定 operator_config"
            )
        self.operator_pool = build_operator_pool(
            operator_config,
            consumer="Bayesian",
            use_defaults=mode == "free",
        )
        self.operator_config = self.operator_pool.as_sample_kwargs()

    def optimize(
        self,
        df_lazy: pl.LazyFrame,
        *,
        n_trials: int | None = None,
        n_jobs: int | None = None,
        top_n: int = 100,
        study_name: str | None = None,
        storage: str | None = None,
    ) -> pl.DataFrame:
        available = set(df_lazy.collect_schema().names())
        required = {
            "trade_time", "code", self.return_column, *self.feature_names,
        }
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"输入数据缺少列: {missing}")
        if top_n < 1:
            raise ValueError("top_n 必须为正整数")
        evaluation_type = detect_evaluation_type(df_lazy)
        ensure_evaluation_supported(evaluation_type)

        values = dict(self.config.__dict__)
        if n_trials is not None:
            values["n_trials"] = n_trials
        if n_jobs is not None:
            values["n_jobs"] = n_jobs
        config = BayesianConfig(**values)
        engine = BayesianEngine(
            df_lazy=df_lazy,
            feature_names=self.feature_names,
            return_column=self.return_column,
            mode=self.mode,
            config=config,
            operator_config=self.operator_config,
            evaluation_type=evaluation_type,
            evaluation_config=self.evaluation_config,
        )
        records = engine.train(study_name=study_name, storage=storage)
        if not records:
            raise RuntimeError(
                "没有成功完成的 trial；请查看上方 Optuna trial 错误日志"
            )
        return (
            pl.DataFrame(records)
            .sort("score", descending=True, nulls_last=True)
            .unique("formula_id", keep="first", maintain_order=True)
            .head(top_n)
        )

__all__ = ["Launcher"]

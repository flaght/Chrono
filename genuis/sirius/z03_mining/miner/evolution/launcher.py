"""进化算法因子挖掘公开入口。"""

from collections.abc import Iterable, Mapping, Sequence

import polars as pl

from miner.evaluate import TimeSeriesEvaluationConfig, detect_evaluation_type, ensure_evaluation_supported
from miner.formula import MiningMode, validate_names
from miner.operators import build_operator_pool

from .config import EvolutionConfig
from .engine import EvolutionEngine


class Launcher:
    """提供与 Bayesian、AlphaGPT 后端一致的 optimize 接口。"""

    def __init__(
        self,
        *,
        feature_names: Iterable[str],
        return_column: str,
        mode: MiningMode = "free",
        config: EvolutionConfig = None,
        operator_config: Mapping[str, Sequence[str]] = None,
        evaluation_config: TimeSeriesEvaluationConfig | None = None,
    ) -> None:
        self.feature_names = list(validate_names(feature_names, "feature_names"))
        self.return_column = return_column
        self.mode = mode
        self.config = config or EvolutionConfig()
        self.evaluation_config = evaluation_config or TimeSeriesEvaluationConfig()

        if mode not in ("free", "directed"):
            raise ValueError("mode 必须为 'free' 或 'directed'")
        if mode == "directed" and operator_config is None:
            raise ValueError(
                "定向挖掘必须显式指定 operator_config"
            )
        self.operators = build_operator_pool(
            operator_config,
            consumer="Evolution",
            use_defaults=mode == "free",
        )

    def optimize(
        self,
        df_lazy: pl.LazyFrame,
        *,
        generations: int  = None,
        population_size: int = None,
        top_n: int = 100,
    ) -> pl.DataFrame:
        available = set(df_lazy.collect_schema().names())
        required = {"trade_time", "code", self.return_column, *self.feature_names}
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"输入数据缺少列: {missing}")
        if top_n < 1:
            raise ValueError("top_n 必须为正整数")
        evaluation_type = detect_evaluation_type(df_lazy)
        ensure_evaluation_supported(evaluation_type)

        values = dict(self.config.__dict__)
        if generations is not None:
            values["generations"] = generations
        if population_size is not None:
            values["population_size"] = population_size
            values["elite_size"] = min(
                values["elite_size"], population_size - 1
            )
            values["tournament_size"] = min(
                values["tournament_size"], population_size
            )
        config = EvolutionConfig(**values)
        engine = EvolutionEngine(
            df_lazy=df_lazy,
            feature_names=self.feature_names,
            return_column=self.return_column,
            config=config,
            operators=self.operators,
            evaluation_type=evaluation_type,
            evaluation_config=self.evaluation_config,
        )
        records = engine.train()
        if not records:
            return pl.DataFrame(
                schema={
                    "generation": pl.Int64,
                    "score": pl.Float64,
                    "formula_id": pl.String,
                    "formula": pl.String,
                }
            )
        return (
            pl.DataFrame(records)
            .sort("score", descending=True, nulls_last=True)
            .head(top_n)
        )


__all__ = ["Launcher"]

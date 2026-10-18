"""基于 Polars 的 AlphaGPT 因子挖掘公开入口。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import polars as pl

from miner.evaluate import TimeSeriesEvaluationConfig, detect_evaluation_type, ensure_evaluation_supported
from miner.formula import MiningMode, validate_names
from miner.operators import build_operator_pool

from .model import AlphaGPT
from .config import AlphaGPTConfig
from .engine import AlphaEngine
from .ops import build_operator_specs


class Launcher:
    """提供与其他挖掘后端一致的 ``Launcher.optimize`` 调用方式。"""

    def __init__(
        self,
        *,
        feature_names: Iterable[str],
        return_column: str,
        mode: MiningMode = "free",
        config: AlphaGPTConfig | None = None,
        operator_config: Mapping[str, Sequence[str]] | None = None,
        evaluation_config: TimeSeriesEvaluationConfig | None = None,
    ) -> None:
        self.feature_names = validate_names(feature_names, "feature_names")
        self.return_column = return_column
        self.mode = mode
        self.config = config or AlphaGPTConfig()
        self.evaluation_config = evaluation_config or TimeSeriesEvaluationConfig()

        if mode not in ("free", "directed"):
            raise ValueError("mode 必须为 'free' 或 'directed'")
        if mode == "directed" and operator_config is None:
            raise ValueError(
                "定向挖掘必须显式指定 operator_config"
            )
        self.operator_pool = build_operator_pool(
            operator_config,
            consumer="AlphaGPT",
            use_defaults=mode == "free",
        )
        self.operator_specs = build_operator_specs(
            self.config.periods, self.operator_pool
        )

    def optimize(
        self,
        df_lazy: pl.LazyFrame,
        *,
        train_steps: int | None = None,
        batch_size: int | None = None,
        top_n: int = 100,
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

        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "AlphaGPT 挖掘器需要 PyTorch: pip install torch"
            ) from error

        values = dict(self.config.__dict__)
        if train_steps is not None:
            values["train_steps"] = train_steps
        if batch_size is not None:
            values["batch_size"] = batch_size
        config = AlphaGPTConfig(**values)
        device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        operator_count = len(self.operator_specs)
        if config.verbose:
            print(
                "[AlphaGPT] 开始挖掘："
                f"设备={device}，训练轮数={config.train_steps}，"
                f"批量={config.batch_size}，公式长度={config.max_formula_len}，"
                f"基础特征={len(self.feature_names)}，算子Token={operator_count}，"
                f"目标列={self.return_column}，模式={self.mode}，"
                f"特征={self.feature_names}",
                flush=True,
            )
        model = AlphaGPT(
            vocab_size=len(self.feature_names) + operator_count,
            max_formula_len=config.max_formula_len,
        ).to(device)
        engine = AlphaEngine(
            model=model,
            df_lazy=df_lazy,
            feature_names=self.feature_names,
            return_column=self.return_column,
            operator_specs=self.operator_specs,
            config=config,
            device=device,
            evaluation_type=evaluation_type,
            evaluation_config=self.evaluation_config,
        )
        records = engine.train()
        if config.verbose:
            print(
                f"[AlphaGPT] 挖掘结束：有效候选={len(records)}，"
                f"返回前{min(top_n, len(records))}个因子",
                flush=True,
            )
        if not records:
            return pl.DataFrame(
                schema={
                    "step": pl.Int64,
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

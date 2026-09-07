"""按输入数据类型分派因子评估器。"""

import math
from dataclasses import dataclass
from numbers import Number
from typing import Literal

import polars as pl

from .formula import Formula, FormulaCompiler

EvaluationType = Literal["time_series", "cross_section"]


@dataclass(frozen=True)
class TimeSeriesEvaluationConfig:
    """``cux001`` 时序因子评估参数。"""

    resampling_win: int = 1
    roll_win: int = 252
    fee: float = 0.0003
    scale_method: str = "roll_min_max"
    annualization_factor: int = 252
    is_check: bool = False


def detect_evaluation_type(
    df_lazy: pl.LazyFrame, *, code_column: str = "code"
) -> EvaluationType:
    """根据品种数量识别单标时序数据或多标截面数据。"""
    codes = df_lazy.select(
        pl.col(code_column).drop_nulls().unique().head(2).alias(code_column)
    ).collect()
    if codes.height == 0:
        raise ValueError(f"输入列 {code_column!r} 没有有效品种值")
    return "time_series" if codes.height == 1 else "cross_section"


def ensure_evaluation_supported(evaluation_type: EvaluationType) -> None:
    """在正式挖掘前阻止尚无评估实现的截面数据。"""
    if evaluation_type == "cross_section":
        raise NotImplementedError(
            "检测到多标截面数据；截面因子评估器尚未实现，"
            "本次挖掘已停止，不会使用时序评估器或旧的截面 Rank IC 评估。"
        )


def _empty_metrics(observations: int = 0) -> dict[str, float | int]:
    """生成样本不足时的兼容评分字段。"""
    return {
        "ic_mean": 0.0, "abs_ic_mean": 0.0, "ic_sharpe": 0.0,
        "observations": observations, "period_count": 0,
    }


def _to_python_number(value: object) -> object:
    """将 NumPy 等数值标量转换为可序列化的 Python 标量。"""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, Number):
        return float(value)
    return value


def evaluate_formula(
    df_lazy: pl.LazyFrame,
    formula: Formula,
    *,
    return_column: str,
    evaluation_type: EvaluationType | None = None,
    min_observations: int = 100,
    compiler: FormulaCompiler | None = None,
    time_series_config: TimeSeriesEvaluationConfig | None = None,
) -> dict[str, object]:
    """评估候选公式；目前仅实现单标时序因子评估。"""
    evaluation_type = evaluation_type or detect_evaluation_type(df_lazy)
    ensure_evaluation_supported(evaluation_type)
    if evaluation_type != "time_series":
        raise ValueError(f"未知评估类型: {evaluation_type}")

    config = time_series_config or TimeSeriesEvaluationConfig()
    compiler = compiler or FormulaCompiler()
    keys = list(compiler.key_columns)
    factor = compiler.compile(df_lazy, formula, output_name="_factor")
    prepared = (
        factor.join(df_lazy.select([*keys, return_column]), on=keys, how="inner")
        .select(
            pl.col(keys[0]).alias("trade_time"),
            pl.col("_factor"),
            pl.col(return_column).alias("_return"),
        )
        .drop_nulls(["trade_time", "_factor", "_return"])
        .filter(pl.col("_factor").is_finite() & pl.col("_return").is_finite())
        .collect()
    )
    observations = prepared.height
    if observations < min_observations:
        return _empty_metrics(observations)

    # 延迟导入，避免仅使用公式模块时加载绘图等评估依赖。
    from evaluate.cux001 import FactorEvaluatePolars

    evaluator = FactorEvaluatePolars(
        prepared,
        resampling_win=config.resampling_win,
        factor_name="_factor",
        ret_name="_return",
        roll_win=config.roll_win,
        fee=config.fee,
        scale_method=config.scale_method,
        annualization_factor=config.annualization_factor,
        expression=formula.text,
        name=formula.factor_id,
    )
    raw_stats = evaluator.run(is_check=config.is_check)
    stats = {key: _to_python_number(value) for key, value in raw_stats.items()}
    ic_mean = float(stats.get("ic_mean") or 0.0)
    ic_ir = float(stats.get("ic_ir") or 0.0)
    if not math.isfinite(ic_mean):
        ic_mean = 0.0
    if not math.isfinite(ic_ir):
        ic_ir = 0.0
    stats.update({
        "ic_mean": ic_mean,
        "abs_ic_mean": abs(ic_mean),
        "ic_sharpe": ic_ir,
        "observations": observations,
        "period_count": (
            evaluator.resample_data_pl.height
            if evaluator.resample_data_pl is not None else 0
        ),
    })
    return stats


__all__ = [
    "EvaluationType", "TimeSeriesEvaluationConfig",
    "detect_evaluation_type", "ensure_evaluation_supported", "evaluate_formula",
]

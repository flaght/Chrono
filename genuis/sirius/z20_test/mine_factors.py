from collections.abc import Iterable

import polars as pl

from feature.utils import formula as ops


DEFAULT_PERIODS = (5, 10, 20)
DEFAULT_OPERATORS = ("ts_mean", "ts_std", "ts_rank")


def evaluate_candidates(
    candidate_lazy: pl.LazyFrame,
    candidate_names: Iterable[str],
) -> pl.DataFrame:
    """按候选值与下一周期收益的全样本相关系数排序。"""
    candidate_names = tuple(candidate_names)
    data = candidate_lazy.collect()

    correlations = data.select(
        [
            pl.corr(pl.col(name), pl.col("_forward_return")).alias(name)
            for name in candidate_names
        ]
    ).row(0, named=True)

    return (
        pl.DataFrame(
            {
                "factor": candidate_names,
                "correlation": [correlations[name] for name in candidate_names],
            }
        )
        .with_columns(
            pl.col("correlation").abs().alias("abs_correlation")
        )
        .sort("abs_correlation", descending=True, nulls_last=True)
    )


def build_candidates(
    df_lazy: pl.LazyFrame,
    *,
    periods: Iterable[int] = DEFAULT_PERIODS,
    operator_names: Iterable[str] = DEFAULT_OPERATORS,
    forward_period: int = 1,
) -> tuple[pl.LazyFrame, tuple[str, ...]]:
    """生成候选因子图，并返回候选列名。"""
    periods = tuple(periods)
    operator_names = tuple(operator_names)

    # 第一层：基础派生结果先落列，下一层窗口算子只引用这些列。
    result = (
        df_lazy
        .sort(["trade_time", "code"])
        .with_columns(
            ops.log_return("close").alias("_return"),
            ops.pct_change("volume").alias("_volume_change"),
            ops.pct_change("openint").alias("_openint_change"),
            ops.safe_div(
                pl.col("high") - pl.col("low"),
                "close",
            ).alias("_range_ratio"),
        )
    )
    base_columns = (
        "_return",
        "_volume_change",
        "_openint_change",
        "_range_ratio",
    )
    candidate_expressions: list[pl.Expr] = []
    candidate_names: list[str] = []

    # 第二层：枚举“基础结果 × 时序算子 × 周期”。
    for base_column in base_columns:
        base_name = base_column.removeprefix("_")
        for operator_name in operator_names:
            operator = ops.OPERATORS[operator_name]
            for period in periods:
                candidate_name = f"{operator_name}_{base_name}_{period}"
                candidate_expressions.append(
                    operator(base_column, period).alias(candidate_name)
                )
                candidate_names.append(candidate_name)
    
    # 额外枚举两个双变量相关性候选。
    for right_column in ("_volume_change", "_openint_change"):
        right_name = right_column.removeprefix("_")
        for period in periods:
            candidate_name = f"ts_corr_return_{right_name}_{period}"
            candidate_expressions.append(
                ops.ts_corr("_return", right_column, period).alias(
                    candidate_name
                )
            )
            candidate_names.append(candidate_name)

    result = result.with_columns(candidate_expressions)

    # 评估目标独立生成；负 shift 只允许出现在挖掘评估中，不能进入正式因子。
    result = result.with_columns(
        (
            ops.safe_div(
                pl.col("close").shift(-forward_period).over("code"),
                "close",
            )
            - 1.0
        ).alias("_forward_return")
    )

    return (
        result.select(
            ["trade_time", "code", "_forward_return", *candidate_names]
        ),
        tuple(candidate_names),
    )


def mine_file(
    file_path: str,
    *,
    periods: Iterable[int] = DEFAULT_PERIODS,
    top_n: int = 20,
) -> pl.DataFrame:
    """懒加载 IPC/Feather 文件并返回排名靠前的候选表达式。"""
    df_lazy = pl.scan_ipc(file_path)
    candidate_lazy, candidate_names = build_candidates(
        df_lazy,
        periods=periods,
    )
    return evaluate_candidates(candidate_lazy, candidate_names).head(top_n)


if __name__ == "__main__":
    FILE_PATH = (
        "/workspace/worker/pj/Chrono/genuis/mizar/records/"
        "ricso2/rbb/basic/recent_data.feather"
    )
    ranking = mine_file(FILE_PATH, periods=(5, 10, 20), top_n=20)
    print(ranking)

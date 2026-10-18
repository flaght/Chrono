"""
因子代号: tc005_036
原历史名: tc005_036
因子定义: 多窗口极差比值的滑动窗口排序分位因子，衡量多尺度极端波动
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIODS = (5, 15, 15)
NAME = "tc005_036"


def calculate(df_lazy: pl.LazyFrame, periods: tuple[int, int, int]) -> pl.LazyFrame:
    """使用trade_time、code、high、low计算 tc005_036 核心公式。"""
    (fast, slow, rank_period) = periods
    return (
        df_lazy
        .with_columns(
            ((pl.col('high')).rolling_max(fast).over("code")).alias("_cr045_stage_0")
        )
        .with_columns(
            ((pl.col('low')).rolling_min(fast).over("code")).alias("_cr045_stage_1")
        )
        .with_columns(
            ((pl.col("_cr045_stage_0")) - (pl.col("_cr045_stage_1"))).alias("_cr045_stage_2")
        )
        .with_columns(
            ((pl.col('high')).rolling_max(slow).over("code")).alias("_cr045_stage_3")
        )
        .with_columns(
            ((pl.col('low')).rolling_min(slow).over("code")).alias("_cr045_stage_4")
        )
        .with_columns(
            ((pl.col("_cr045_stage_3")) - (pl.col("_cr045_stage_4"))).alias("_cr045_stage_5")
        )
        .with_columns(
            (pl.when((pl.col("_cr045_stage_5")).is_not_null() & ((pl.col("_cr045_stage_5")) != 0)).then((pl.col("_cr045_stage_2")) / (pl.col("_cr045_stage_5"))).otherwise(None)).alias("_cr045_stage_6")
        )
        .with_columns(
            (rolling_rank(pl.col("_cr045_stage_6"), rank_period, pct=True).over("code")).alias("_cr045_stage_7")
        )
        .with_columns(
            (pl.col("_cr045_stage_7")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, periods: tuple[int, int, int] = DEFAULT_PERIODS) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc005_036；输入必须包含trade_time、code、high、low。"""
    if len(periods) != 3:
        raise ValueError("periods 必须包含 fast、slow、rank_period")
    fast, slow, rank_period = periods
    validate_period(fast)
    validate_period(slow)
    validate_period(rank_period)
    if fast >= slow:
        raise ValueError("fast 必须小于 slow")
    return calculate(df_lazy.sort(["trade_time", "code"]), periods).select(["trade_time", "code", NAME])


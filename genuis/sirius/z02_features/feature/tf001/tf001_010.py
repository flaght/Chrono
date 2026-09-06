"""
因子代号: tf001_010
原历史名: tf001_010
因子定义: 持仓非流动性变异系数：非流动性指标在 period 周期的滚动变异系数 (std / mean)
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_010"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、close、openint 计算 tf001_010 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).shift(1).over("code")).alias("_oi011_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_oi011_stage_0")).is_not_null() & ((pl.col("_oi011_stage_0")) != 0)).then((pl.col('close')) / (pl.col("_oi011_stage_0"))).otherwise(None)).alias("_oi011_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_oi011_stage_1")) > 0).then((pl.col("_oi011_stage_1")).log()).otherwise(None)).alias("_oi011_stage_2")
        )
        .with_columns(
            (pl.when((pl.lit(10000000.0)).is_not_null() & ((pl.lit(10000000.0)) != 0)).then((pl.col('openint')) / (pl.lit(10000000.0))).otherwise(None)).alias("_oi011_stage_3")
        )
        .with_columns(
            (pl.when((pl.col("_oi011_stage_3")).is_not_null() & ((pl.col("_oi011_stage_3")) != 0)).then((pl.col("_oi011_stage_2")) / (pl.col("_oi011_stage_3"))).otherwise(None)).alias("_oi011_stage_4")
        )
        .with_columns(
            ((pl.col("_oi011_stage_4")).rolling_std(period).over("code")).alias("_oi011_stage_5")
        )
        .with_columns(
            ((pl.col("_oi011_stage_4")).rolling_mean(period).over("code")).alias("_oi011_stage_6")
        )
        .with_columns(
            (pl.when((pl.col("_oi011_stage_6")).is_not_null() & ((pl.col("_oi011_stage_6")) != 0)).then((pl.col("_oi011_stage_5")) / (pl.col("_oi011_stage_6"))).otherwise(None)).alias("_oi011_stage_7")
        )
        .with_columns(
            (pl.col("_oi011_stage_7")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_010；输入必须包含 trade_time、code、close、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


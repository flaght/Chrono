"""
因子代号: tf001_044
原历史名: tf001_044
因子定义: 一致性交易强度：价量仓同向变动的一致性综合得分
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_044"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、close、open、high、low、openint 计算 tf001_044 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')) - (pl.col('open'))).alias("_oi046_stage_0")
        )
        .with_columns(
            ((pl.col("_oi046_stage_0")).abs()).alias("_oi046_stage_1")
        )
        .with_columns(
            ((pl.col('high')) - (pl.col('low'))).alias("_oi046_stage_2")
        )
        .with_columns(
            ((pl.lit(0.5)) * (pl.col("_oi046_stage_2"))).alias("_oi046_stage_3")
        )
        .with_columns(
            ((pl.col("_oi046_stage_1")) <= (pl.col("_oi046_stage_3"))).alias("_oi046_stage_4")
        )
        .with_columns(
            (pl.when(pl.col("_oi046_stage_4")).then(pl.lit(1.0)).otherwise(pl.lit(0.0))).alias("_oi046_stage_5")
        )
        .with_columns(
            ((pl.col('openint')) * (pl.col("_oi046_stage_5"))).alias("_oi046_stage_6")
        )
        .with_columns(
            ((pl.col("_oi046_stage_6")).rolling_sum(period).over("code")).alias("_oi046_stage_7")
        )
        .with_columns(
            ((pl.col('openint')).rolling_sum(period).over("code")).alias("_oi046_stage_8")
        )
        .with_columns(
            (pl.when((pl.col("_oi046_stage_8")).is_not_null() & ((pl.col("_oi046_stage_8")) != 0)).then((pl.col("_oi046_stage_7")) / (pl.col("_oi046_stage_8"))).otherwise(None)).alias("_oi046_stage_9")
        )
        .with_columns(
            (pl.col("_oi046_stage_9")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_044；输入必须包含 trade_time、code、close、open、high、low、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


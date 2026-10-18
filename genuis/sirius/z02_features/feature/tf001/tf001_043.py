"""
因子代号: tf001_043
原历史名: tf001_043
因子定义: 一致性增仓买入交易：价格上涨且持仓量显著增加的共振强度
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_043"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、close、open、high、low、openint 计算 tf001_043 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')) > (pl.col('open'))).alias("_oi045_stage_0")
        )
        .with_columns(
            (pl.when(pl.col("_oi045_stage_0")).then(pl.lit(1.0)).otherwise(pl.lit(0.0))).alias("_oi045_stage_1")
        )
        .with_columns(
            ((pl.col('openint')) * (pl.col("_oi045_stage_1"))).alias("_oi045_stage_2")
        )
        .with_columns(
            ((pl.col('close')) - (pl.col('open'))).alias("_oi045_stage_3")
        )
        .with_columns(
            ((pl.col("_oi045_stage_3")).abs()).alias("_oi045_stage_4")
        )
        .with_columns(
            ((pl.col('high')) - (pl.col('low'))).alias("_oi045_stage_5")
        )
        .with_columns(
            ((pl.lit(0.5)) * (pl.col("_oi045_stage_5"))).alias("_oi045_stage_6")
        )
        .with_columns(
            ((pl.col("_oi045_stage_4")) <= (pl.col("_oi045_stage_6"))).alias("_oi045_stage_7")
        )
        .with_columns(
            (pl.when(pl.col("_oi045_stage_7")).then(pl.lit(1.0)).otherwise(pl.lit(0.0))).alias("_oi045_stage_8")
        )
        .with_columns(
            ((pl.col("_oi045_stage_2")) * (pl.col("_oi045_stage_8"))).alias("_oi045_stage_9")
        )
        .with_columns(
            ((pl.col("_oi045_stage_9")).rolling_sum(period).over("code")).alias("_oi045_stage_10")
        )
        .with_columns(
            ((pl.col('openint')).rolling_sum(period).over("code")).alias("_oi045_stage_11")
        )
        .with_columns(
            (pl.when((pl.col("_oi045_stage_11")).is_not_null() & ((pl.col("_oi045_stage_11")) != 0)).then((pl.col("_oi045_stage_10")) / (pl.col("_oi045_stage_11"))).otherwise(None)).alias("_oi045_stage_12")
        )
        .with_columns(
            (pl.col("_oi045_stage_12")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_043；输入必须包含 trade_time、code、close、open、high、low、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


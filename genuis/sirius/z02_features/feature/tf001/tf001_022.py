"""
因子代号: tf001_022
原历史名: tf001_022
因子定义: 持仓量加权收益率标准差：以持仓量为权重的收益率加权波动率
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_022"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、open、high、low、close、openint 计算 tf001_022 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')) - (pl.col('open'))).alias("_oi023_stage_0")
        )
        .with_columns(
            ((pl.col("_oi023_stage_0")).abs()).alias("_oi023_stage_1")
        )
        .with_columns(
            ((pl.col('high')) - (pl.col('low'))).alias("_oi023_stage_2")
        )
        .with_columns(
            ((pl.col("_oi023_stage_2")).abs()).alias("_oi023_stage_3")
        )
        .with_columns(
            ((pl.col("_oi023_stage_1")) <= (pl.col("_oi023_stage_3"))).alias("_oi023_stage_4")
        )
        .with_columns(
            ((pl.col('close')) <= (pl.col('open'))).alias("_oi023_stage_5")
        )
        .with_columns(
            ((pl.col("_oi023_stage_4")) & (pl.col("_oi023_stage_5"))).alias("_oi023_stage_6")
        )
        .with_columns(
            (pl.when(pl.col("_oi023_stage_6")).then(pl.col('openint')).otherwise(pl.lit(0.0))).alias("_oi023_stage_7")
        )
        .with_columns(
            ((pl.col("_oi023_stage_7")).rolling_sum(period).over("code")).alias("_oi023_stage_8")
        )
        .with_columns(
            ((pl.col('openint')).rolling_sum(period).over("code")).alias("_oi023_stage_9")
        )
        .with_columns(
            (pl.when((pl.col("_oi023_stage_9")).is_not_null() & ((pl.col("_oi023_stage_9")) != 0)).then((pl.col("_oi023_stage_8")) / (pl.col("_oi023_stage_9"))).otherwise(None)).alias("_oi023_stage_10")
        )
        .with_columns(
            (pl.col("_oi023_stage_10")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_022；输入必须包含 trade_time、code、open、high、low、close、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


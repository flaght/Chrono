"""
因子代号: tf001_034
原历史名: tf001_034
因子定义: 持仓量易变指标 (EMO)：价格中心移动相对持仓量的单位变化比率
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_034"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、high、low、openint 计算 tf001_034 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('high')) + (pl.col('low'))).alias("_oi035_stage_0")
        )
        .with_columns(
            ((pl.lit(0.5)) * (pl.col("_oi035_stage_0"))).alias("_oi035_stage_1")
        )
        .with_columns(
            ((pl.col('high')).shift(1).over("code")).alias("_oi035_stage_2")
        )
        .with_columns(
            ((pl.col('low')).shift(1).over("code")).alias("_oi035_stage_3")
        )
        .with_columns(
            ((pl.col("_oi035_stage_2")) + (pl.col("_oi035_stage_3"))).alias("_oi035_stage_4")
        )
        .with_columns(
            ((pl.lit(0.5)) * (pl.col("_oi035_stage_4"))).alias("_oi035_stage_5")
        )
        .with_columns(
            ((pl.col("_oi035_stage_1")) - (pl.col("_oi035_stage_5"))).alias("_oi035_stage_6")
        )
        .with_columns(
            (pl.when((pl.lit(10000000.0)).is_not_null() & ((pl.lit(10000000.0)) != 0)).then((pl.col('openint')) / (pl.lit(10000000.0))).otherwise(None)).alias("_oi035_stage_7")
        )
        .with_columns(
            ((pl.col('high')) - (pl.col('low'))).alias("_oi035_stage_8")
        )
        .with_columns(
            (pl.when((pl.col("_oi035_stage_8")).is_not_null() & ((pl.col("_oi035_stage_8")) != 0)).then((pl.col("_oi035_stage_7")) / (pl.col("_oi035_stage_8"))).otherwise(None)).alias("_oi035_stage_9")
        )
        .with_columns(
            (pl.when((pl.col("_oi035_stage_9")).is_not_null() & ((pl.col("_oi035_stage_9")) != 0)).then((pl.col("_oi035_stage_6")) / (pl.col("_oi035_stage_9"))).otherwise(None)).alias("_oi035_stage_10")
        )
        .with_columns(
            ((pl.col("_oi035_stage_10")).rolling_mean(period).over("code")).alias("_oi035_stage_11")
        )
        .with_columns(
            (pl.col("_oi035_stage_11")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_034；输入必须包含 trade_time、code、high、low、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


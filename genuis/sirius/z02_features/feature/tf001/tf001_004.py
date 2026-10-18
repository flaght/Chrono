"""
因子代号: tf001_004
原历史名: tf001_004
因子定义: 持仓量非流动性指标 3：收益率与持仓量变化率的比值
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_004"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、close、openint 计算 tf001_004 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).shift(1).over("code")).alias("_oi004_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_oi004_stage_0")).is_not_null() & ((pl.col("_oi004_stage_0")) != 0)).then((pl.col('close')) / (pl.col("_oi004_stage_0"))).otherwise(None)).alias("_oi004_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_oi004_stage_1")) > 0).then((pl.col("_oi004_stage_1")).log()).otherwise(None)).alias("_oi004_stage_2")
        )
        .with_columns(
            ((pl.col("_oi004_stage_2")).rolling_mean(period).over("code")).alias("_oi004_stage_3")
        )
        .with_columns(
            (pl.when((pl.lit(1000000.0)).is_not_null() & ((pl.lit(1000000.0)) != 0)).then((pl.col('openint')) / (pl.lit(1000000.0))).otherwise(None)).alias("_oi004_stage_4")
        )
        .with_columns(
            ((pl.col("_oi004_stage_4")).rolling_mean(period).over("code")).alias("_oi004_stage_5")
        )
        .with_columns(
            (pl.when((pl.col("_oi004_stage_5")).is_not_null() & ((pl.col("_oi004_stage_5")) != 0)).then((pl.col("_oi004_stage_3")) / (pl.col("_oi004_stage_5"))).otherwise(None)).alias("_oi004_stage_6")
        )
        .with_columns(
            ((pl.col("_oi004_stage_2")).rolling_std(period).over("code")).alias("_oi004_stage_7")
        )
        .with_columns(
            ((pl.col("_oi004_stage_4")).rolling_std(period).over("code")).alias("_oi004_stage_8")
        )
        .with_columns(
            (pl.rolling_corr(pl.col("_oi004_stage_7"), pl.col("_oi004_stage_8"), window_size=period).over("code")).alias("_oi004_stage_9")
        )
        .with_columns(
            ((pl.col("_oi004_stage_9")) * (pl.col("_oi004_stage_7"))).alias("_oi004_stage_10")
        )
        .with_columns(
            (pl.when((pl.col("_oi004_stage_8")).is_not_null() & ((pl.col("_oi004_stage_8")) != 0)).then((pl.col("_oi004_stage_10")) / (pl.col("_oi004_stage_8"))).otherwise(None)).alias("_oi004_stage_11")
        )
        .with_columns(
            (pl.when((pl.col("_oi004_stage_5")).is_not_null() & ((pl.col("_oi004_stage_5")) != 0)).then((pl.col("_oi004_stage_8")) / (pl.col("_oi004_stage_5"))).otherwise(None)).alias("_oi004_stage_12")
        )
        .with_columns(
            ((pl.col("_oi004_stage_12")) ** (pl.lit(2))).alias("_oi004_stage_13")
        )
        .with_columns(
            ((pl.col("_oi004_stage_11")) * (pl.col("_oi004_stage_13"))).alias("_oi004_stage_14")
        )
        .with_columns(
            ((pl.col("_oi004_stage_6")) + (pl.col("_oi004_stage_14"))).alias("_oi004_stage_15")
        )
        .with_columns(
            (pl.col("_oi004_stage_15")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_004；输入必须包含 trade_time、code、close、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


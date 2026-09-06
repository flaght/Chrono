"""
因子代号: tf001_009
原历史名: tf001_009
因子定义: 持仓量波峰计数：持仓量超过 (均值 + 1倍标准差) 的局部峰值周期数
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_009"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、openint 计算 tf001_009 核心公式。"""
    return (
        df_lazy
        .with_columns(
            (pl.when((pl.lit(1000000.0)).is_not_null() & ((pl.lit(1000000.0)) != 0)).then((pl.col('openint')) / (pl.lit(1000000.0))).otherwise(None)).alias("_oi010_stage_0")
        )
        .with_columns(
            ((pl.col("_oi010_stage_0")).rolling_mean(period).over("code")).alias("_oi010_stage_1")
        )
        .with_columns(
            ((pl.col("_oi010_stage_0")).rolling_std(period).over("code")).alias("_oi010_stage_2")
        )
        .with_columns(
            ((pl.col("_oi010_stage_1")) + (pl.col("_oi010_stage_2"))).alias("_oi010_stage_3")
        )
        .with_columns(
            ((pl.col("_oi010_stage_0")) > (pl.col("_oi010_stage_3"))).alias("_oi010_stage_4")
        )
        .with_columns(
            (pl.when(pl.col("_oi010_stage_4")).then(pl.lit(1.0)).otherwise(pl.lit(0.0))).alias("_oi010_stage_5")
        )
        .with_columns(
            ((pl.col("_oi010_stage_0")) + (pl.col("_oi010_stage_5"))).alias("_oi010_stage_6")
        )
        .with_columns(
            ((pl.col("_oi010_stage_6")).rolling_sum(period).over("code")).alias("_oi010_stage_7")
        )
        .with_columns(
            (pl.col("_oi010_stage_7")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_009；输入必须包含 trade_time、code、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


"""
因子代号: tf001_002
原历史名: tf001_002
因子定义: 持仓量波动率符号加权与价格波动率的相关系数及波动比率
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_002"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、close、openint 计算 tf001_002 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).rolling_std(period).over("code")).alias("_oi002_stage_0")
        )
        .with_columns(
            (pl.when((pl.lit(1000000.0)).is_not_null() & ((pl.lit(1000000.0)) != 0)).then((pl.col('openint')) / (pl.lit(1000000.0))).otherwise(None)).alias("_oi002_stage_1")
        )
        .with_columns(
            ((pl.col("_oi002_stage_1")).rolling_std(period).over("code")).alias("_oi002_stage_2")
        )
        .with_columns(
            (pl.rolling_corr(pl.col("_oi002_stage_0"), pl.col("_oi002_stage_2"), window_size=period).over("code")).alias("_oi002_stage_3")
        )
        .with_columns(
            ((pl.col("_oi002_stage_0")).rolling_std(period).over("code")).alias("_oi002_stage_4")
        )
        .with_columns(
            ((pl.col("_oi002_stage_3")) * (pl.col("_oi002_stage_4"))).alias("_oi002_stage_5")
        )
        .with_columns(
            ((pl.col("_oi002_stage_2")).rolling_std(period).over("code")).alias("_oi002_stage_6")
        )
        .with_columns(
            (pl.when((pl.col("_oi002_stage_6")).is_not_null() & ((pl.col("_oi002_stage_6")) != 0)).then((pl.col("_oi002_stage_5")) / (pl.col("_oi002_stage_6"))).otherwise(None)).alias("_oi002_stage_7")
        )
        .with_columns(
            (pl.col("_oi002_stage_7")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_002；输入必须包含 trade_time、code、close、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


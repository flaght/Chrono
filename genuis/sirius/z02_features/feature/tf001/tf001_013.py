"""
因子代号: tf001_013
原历史名: tf001_013
因子定义: 持仓量加权收盘价偏度：以持仓量为权重的收盘价偏离偏度
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_013"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、close、openint 计算 tf001_013 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('openint')).rolling_sum(period).over("code")).alias("_oi014_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_oi014_stage_0")).is_not_null() & ((pl.col("_oi014_stage_0")) != 0)).then((pl.col('openint')) / (pl.col("_oi014_stage_0"))).otherwise(None)).alias("_oi014_stage_1")
        )
        .with_columns(
            ((pl.col('close')) * (pl.col("_oi014_stage_1"))).alias("_oi014_stage_2")
        )
        .with_columns(
            ((pl.col("_oi014_stage_2")).rolling_skew(period).over("code")).alias("_oi014_stage_3")
        )
        .with_columns(
            (pl.col("_oi014_stage_3")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_013；输入必须包含 trade_time、code、close、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


"""
因子代号: tf001_023
原历史名: tf001_023
因子定义: 持仓量变动加速度：持仓量二阶差分相对移动平均的加速强度
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_023"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、value、openint、close、open 计算 tf001_023 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('value')).rolling_sum(period).over("code")).alias("_oi024_stage_0")
        )
        .with_columns(
            ((pl.col('openint')).rolling_sum(period).over("code")).alias("_oi024_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_oi024_stage_1")).is_not_null() & ((pl.col("_oi024_stage_1")) != 0)).then((pl.col("_oi024_stage_0")) / (pl.col("_oi024_stage_1"))).otherwise(None)).alias("_oi024_stage_2")
        )
        .with_columns(
            ((pl.col("_oi024_stage_2")) - (pl.col('open'))).alias("_oi024_stage_3")
        )
        .with_columns(
            ((pl.col('close')).shift(1).over("code")).alias("_oi024_stage_4")
        )
        .with_columns(
            (pl.when((pl.col("_oi024_stage_4")).is_not_null() & ((pl.col("_oi024_stage_4")) != 0)).then((pl.col("_oi024_stage_3")) / (pl.col("_oi024_stage_4"))).otherwise(None)).alias("_oi024_stage_5")
        )
        .with_columns(
            (pl.col("_oi024_stage_5")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_023；输入必须包含 trade_time、code、value、openint、close、open。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


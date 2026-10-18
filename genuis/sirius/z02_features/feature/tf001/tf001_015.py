"""
因子代号: tf001_015
原历史名: tf001_015
因子定义: 高价位持仓量占比：价格处于前20%最高区间内的持仓量占总持仓的比重
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_015"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、close、openint 计算 tf001_015 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).rolling_quantile(0.8, window_size=period).over("code")).alias("_oi016_stage_0")
        )
        .with_columns(
            ((pl.col('close')) >= (pl.col("_oi016_stage_0"))).alias("_oi016_stage_1")
        )
        .with_columns(
            (pl.when(pl.col("_oi016_stage_1")).then(pl.col('openint')).otherwise(pl.lit(None))).alias("_oi016_stage_2")
        )
        .with_columns(
            ((pl.col("_oi016_stage_2")).rolling_sum(period).over("code")).alias("_oi016_stage_3")
        )
        .with_columns(
            ((pl.col('openint')).rolling_sum(period).over("code")).alias("_oi016_stage_4")
        )
        .with_columns(
            (pl.when((pl.col("_oi016_stage_4")).is_not_null() & ((pl.col("_oi016_stage_4")) != 0)).then((pl.col("_oi016_stage_3")) / (pl.col("_oi016_stage_4"))).otherwise(None)).alias("_oi016_stage_5")
        )
        .with_columns(
            (pl.col("_oi016_stage_5")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_015；输入必须包含 trade_time、code、close、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


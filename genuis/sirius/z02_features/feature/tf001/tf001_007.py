"""
因子代号: tf001_007
原历史名: tf001_007
因子定义: 持仓量与价格相关系数：持仓量与收盘价在 period 周期内的滚动相关性
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_007"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、close、openint 计算 tf001_007 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).shift(1).over("code")).alias("_oi008_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_oi008_stage_0")).is_not_null() & ((pl.col("_oi008_stage_0")) != 0)).then((pl.col('close')) / (pl.col("_oi008_stage_0"))).otherwise(None)).alias("_oi008_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_oi008_stage_1")) > 0).then((pl.col("_oi008_stage_1")).log()).otherwise(None)).alias("_oi008_stage_2")
        )
        .with_columns(
            (pl.when((pl.lit(1000000.0)).is_not_null() & ((pl.lit(1000000.0)) != 0)).then((pl.col('openint')) / (pl.lit(1000000.0))).otherwise(None)).alias("_oi008_stage_3")
        )
        .with_columns(
            (pl.rolling_corr(pl.col("_oi008_stage_2"), pl.col("_oi008_stage_3"), window_size=period).over("code")).alias("_oi008_stage_4")
        )
        .with_columns(
            (pl.col("_oi008_stage_4")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_007；输入必须包含 trade_time、code、close、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


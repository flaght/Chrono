"""
因子代号: tf001_024
原历史名: tf001_024
因子定义: 持仓量与振幅相关性：持仓量与日内价格极差 (high - low) 的滚动相关系数
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_024"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、openint、value 计算 tf001_024 核心公式。"""
    return (
        df_lazy
        .with_columns(
            (pl.when((pl.col('openint')).is_not_null() & ((pl.col('openint')) != 0)).then((pl.col('value')) / (pl.col('openint'))).otherwise(None)).alias("_oi025_stage_0")
        )
        .with_columns(
            ((pl.col("_oi025_stage_0")).rolling_mean(period).over("code")).alias("_oi025_stage_1")
        )
        .with_columns(
            ((pl.col("_oi025_stage_1")).rolling_max(period).over("code")).alias("_oi025_stage_2")
        )
        .with_columns(
            ((pl.col('value')).rolling_mean(period).over("code")).alias("_oi025_stage_3")
        )
        .with_columns(
            (pl.rolling_cov(pl.col("_oi025_stage_2"), pl.col("_oi025_stage_3"), window_size=period).over("code")).alias("_oi025_stage_4")
        )
        .with_columns(
            (pl.col("_oi025_stage_4")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_024；输入必须包含 trade_time、code、openint、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


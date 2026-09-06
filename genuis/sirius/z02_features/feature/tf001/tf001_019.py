"""
因子代号: tf001_019
原历史名: tf001_019
因子定义: 对数持仓量厚尾右侧占比：超过90%分位数的对数持仓量占总对数持仓的比重
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_019"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、openint 计算 tf001_019 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('openint')).rolling_quantile(0.9, window_size=period).over("code")).alias("_oi020_stage_0")
        )
        .with_columns(
            ((pl.col('openint')) >= (pl.col("_oi020_stage_0"))).alias("_oi020_stage_1")
        )
        .with_columns(
            (pl.when(pl.col("_oi020_stage_1")).then(pl.col('openint')).otherwise(pl.lit(None))).alias("_oi020_stage_2")
        )
        .with_columns(
            ((pl.col("_oi020_stage_2")).rolling_sum(period).over("code")).alias("_oi020_stage_3")
        )
        .with_columns(
            ((pl.col('openint')).rolling_sum(period).over("code")).alias("_oi020_stage_4")
        )
        .with_columns(
            (pl.when((pl.col("_oi020_stage_4")).is_not_null() & ((pl.col("_oi020_stage_4")) != 0)).then((pl.col("_oi020_stage_3")) / (pl.col("_oi020_stage_4"))).otherwise(None)).alias("_oi020_stage_5")
        )
        .with_columns(
            (pl.col("_oi020_stage_5")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_019；输入必须包含 trade_time、code、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


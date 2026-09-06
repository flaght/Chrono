"""
因子代号: tf001_040
原历史名: tf001_040
因子定义: 持仓量价背离指标：价格创新高而持仓量未创新高的背离程度
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_040"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、value、openint 计算 tf001_040 核心公式。"""
    return (
        df_lazy
        .with_columns(
            (pl.when((pl.col('openint')).is_not_null() & ((pl.col('openint')) != 0)).then((pl.col('value')) / (pl.col('openint'))).otherwise(None)).alias("_oi041_stage_0")
        )
        .with_columns(
            ((pl.col("_oi041_stage_0")).rolling_mean(period).over("code")).alias("_oi041_stage_1")
        )
        .with_columns(
            ((pl.col('value')).rolling_mean(period).over("code")).alias("_oi041_stage_2")
        )
        .with_columns(
            ((pl.col('openint')).rolling_mean(period).over("code")).alias("_oi041_stage_3")
        )
        .with_columns(
            (pl.when((pl.col("_oi041_stage_3")).is_not_null() & ((pl.col("_oi041_stage_3")) != 0)).then((pl.col("_oi041_stage_2")) / (pl.col("_oi041_stage_3"))).otherwise(None)).alias("_oi041_stage_4")
        )
        .with_columns(
            (pl.when((pl.col("_oi041_stage_4")).is_not_null() & ((pl.col("_oi041_stage_4")) != 0)).then((pl.col("_oi041_stage_1")) / (pl.col("_oi041_stage_4"))).otherwise(None)).alias("_oi041_stage_5")
        )
        .with_columns(
            (pl.when((pl.col("_oi041_stage_5")) > 0).then((pl.col("_oi041_stage_5")).log()).otherwise(None)).alias("_oi041_stage_6")
        )
        .with_columns(
            (pl.col("_oi041_stage_6")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_040；输入必须包含 trade_time、code、value、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


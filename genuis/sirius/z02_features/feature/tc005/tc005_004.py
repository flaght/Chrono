"""
因子代号: tc005_004
原历史名: tc005_004
因子定义: N期最高价与最低价区间突破与成交量变化复合因子，衡量价格突破与量能配合
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
NAME = "tc005_004"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用trade_time、code、high、low、volume计算 tc005_004 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('high')).rolling_max(period).over("code")).alias("_cr008_stage_0")
        )
        .with_columns(
            ((pl.col('low')).rolling_min(period).over("code")).alias("_cr008_stage_1")
        )
        .with_columns(
            ((pl.col("_cr008_stage_0")) - (pl.col("_cr008_stage_1"))).alias("_cr008_stage_2")
        )
        .with_columns(
            (pl.when((pl.col("_cr008_stage_1")).is_not_null() & ((pl.col("_cr008_stage_1")) != 0)).then((pl.col("_cr008_stage_2")) / (pl.col("_cr008_stage_1"))).otherwise(None)).alias("_cr008_stage_3")
        )
        .with_columns(
            ((pl.col('volume')).shift(period).over("code")).alias("_cr008_stage_4")
        )
        .with_columns(
            (pl.when((pl.col("_cr008_stage_4")).is_not_null() & ((pl.col("_cr008_stage_4")) != 0)).then((pl.col('volume')) / (pl.col("_cr008_stage_4"))).otherwise(None)).alias("_cr008_stage_5")
        )
        .with_columns(
            ((pl.col("_cr008_stage_5")) - (pl.lit(1))).alias("_cr008_stage_6")
        )
        .with_columns(
            ((pl.col("_cr008_stage_3")) * (pl.col("_cr008_stage_6"))).alias("_cr008_stage_7")
        )
        .with_columns(
            (pl.col("_cr008_stage_7")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc005_004；输入必须包含trade_time、code、high、low、volume。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


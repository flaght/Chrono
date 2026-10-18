"""
因子代号: tc005_009
原历史名: tc005_009
因子定义: N日收盘价收益率的偏度与最高价极差复合因子，衡量收益分布偏斜与高点波动
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
NAME = "tc005_009"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用trade_time、code、close、high计算 tc005_009 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).shift(1).over("code")).alias("_cr014_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_cr014_stage_0")).is_not_null() & ((pl.col("_cr014_stage_0")) != 0)).then((pl.col('close')) / (pl.col("_cr014_stage_0"))).otherwise(None)).alias("_cr014_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_cr014_stage_1")) > 0).then((pl.col("_cr014_stage_1")).log()).otherwise(None)).alias("_cr014_stage_2")
        )
        .with_columns(
            ((pl.col("_cr014_stage_2")).rolling_skew(period).over("code")).alias("_cr014_stage_3")
        )
        .with_columns(
            ((pl.col('high')).rolling_max(period).over("code")).alias("_cr014_stage_4")
        )
        .with_columns(
            ((pl.col('high')).rolling_min(period).over("code")).alias("_cr014_stage_5")
        )
        .with_columns(
            ((pl.col("_cr014_stage_4")) - (pl.col("_cr014_stage_5"))).alias("_cr014_stage_6")
        )
        .with_columns(
            ((pl.col("_cr014_stage_3")) * (pl.col("_cr014_stage_6"))).alias("_cr014_stage_7")
        )
        .with_columns(
            (pl.col("_cr014_stage_7")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc005_009；输入必须包含trade_time、code、close、high。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


"""
因子代号: tc005_011
原历史名: tc005_011
因子定义: N日收盘价与开盘价的对数收益率与最高价极差的协方差复合因子，衡量收益与高点波动的联动性
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
NAME = "tc005_011"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用trade_time、code、close、open、high计算 tc005_011 核心公式。"""
    return (
        df_lazy
        .with_columns(
            (pl.when((pl.col('open')).is_not_null() & ((pl.col('open')) != 0)).then((pl.col('close')) / (pl.col('open'))).otherwise(None)).alias("_cr017_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_cr017_stage_0")) > 0).then((pl.col("_cr017_stage_0")).log()).otherwise(None)).alias("_cr017_stage_1")
        )
        .with_columns(
            ((pl.col('high')).rolling_max(period).over("code")).alias("_cr017_stage_2")
        )
        .with_columns(
            ((pl.col('high')).rolling_min(period).over("code")).alias("_cr017_stage_3")
        )
        .with_columns(
            ((pl.col("_cr017_stage_2")) - (pl.col("_cr017_stage_3"))).alias("_cr017_stage_4")
        )
        .with_columns(
            (pl.rolling_cov(pl.col("_cr017_stage_1"), pl.col("_cr017_stage_4"), window_size=period).over("code")).alias("_cr017_stage_5")
        )
        .with_columns(
            (pl.col("_cr017_stage_5")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc005_011；输入必须包含trade_time、code、close、open、high。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


"""
因子代号: tf001_025
原历史名: tf001_025
因子定义: 持仓量多空失衡度：基于价格涨跌方向推导的主动增减仓失衡指标
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_025"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、open、high、close、openint、value 计算 tf001_025 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).diff(2).over("code")).alias("_oi026_stage_0")
        )
        .with_columns(
            (-(pl.col("_oi026_stage_0"))).alias("_oi026_stage_1")
        )
        .with_columns(
            ((pl.col("_oi026_stage_1")).exp()).alias("_oi026_stage_2")
        )
        .with_columns(
            ((pl.lit(1)) + (pl.col("_oi026_stage_2"))).alias("_oi026_stage_3")
        )
        .with_columns(
            (pl.when((pl.col("_oi026_stage_3")).is_not_null() & ((pl.col("_oi026_stage_3")) != 0)).then((pl.lit(1)) / (pl.col("_oi026_stage_3"))).otherwise(None)).alias("_oi026_stage_4")
        )
        .with_columns(
            ((pl.col('open')).diff(2).over("code")).alias("_oi026_stage_5")
        )
        .with_columns(
            ((pl.col('openint')).diff(2).over("code")).alias("_oi026_stage_6")
        )
        .with_columns(
            (pl.rolling_corr(pl.col("_oi026_stage_5"), pl.col("_oi026_stage_6"), window_size=period).over("code")).alias("_oi026_stage_7")
        )
        .with_columns(
            (pl.rolling_corr(pl.col("_oi026_stage_4"), pl.col("_oi026_stage_7"), window_size=period).over("code")).alias("_oi026_stage_8")
        )
        .with_columns(
            ((pl.col('high')).diff(2).over("code")).alias("_oi026_stage_9")
        )
        .with_columns(
            ((pl.col('value')).diff(2).over("code")).alias("_oi026_stage_10")
        )
        .with_columns(
            (pl.rolling_corr(pl.col("_oi026_stage_9"), pl.col("_oi026_stage_10"), window_size=period).over("code")).alias("_oi026_stage_11")
        )
        .with_columns(
            ((pl.col("_oi026_stage_8")) - (pl.col("_oi026_stage_11"))).alias("_oi026_stage_12")
        )
        .with_columns(
            (pl.col("_oi026_stage_12")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_025；输入必须包含 trade_time、code、open、high、close、openint、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


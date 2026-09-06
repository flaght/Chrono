"""
因子代号: tf001_026
原历史名: tf001_026
因子定义: 持仓量极值价量协同：持仓量极大值与价格极值协同出现的频率指标
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_026"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、high、openint、value 计算 tf001_026 核心公式。"""
    return (
        df_lazy
        .with_columns(
            (pl.when((pl.lit(1000000.0)).is_not_null() & ((pl.lit(1000000.0)) != 0)).then((pl.col('openint')) / (pl.lit(1000000.0))).otherwise(None)).alias("_oi027_stage_0")
        )
        .with_columns(
            (pl.rolling_cov(pl.col("_oi027_stage_0"), pl.col('value'), window_size=period).over("code")).alias("_oi027_stage_1")
        )
        .with_columns(
            (pl.rolling_cov(pl.col('value'), pl.col('high'), window_size=period).over("code")).alias("_oi027_stage_2")
        )
        .with_columns(
            ((pl.col('value')).rolling_std(period).over("code")).alias("_oi027_stage_3")
        )
        .with_columns(
            (pl.when((pl.col("_oi027_stage_3")).is_not_null() & ((pl.col("_oi027_stage_3")) != 0)).then((pl.col("_oi027_stage_2")) / (pl.col("_oi027_stage_3"))).otherwise(None)).alias("_oi027_stage_4")
        )
        .with_columns(
            ((pl.col("_oi027_stage_4")).rolling_mean(period).over("code")).alias("_oi027_stage_5")
        )
        .with_columns(
            (pl.when((pl.col("_oi027_stage_5")).is_not_null() & ((pl.col("_oi027_stage_5")) != 0)).then((pl.col("_oi027_stage_1")) / (pl.col("_oi027_stage_5"))).otherwise(None)).alias("_oi027_stage_6")
        )
        .with_columns(
            (pl.col("_oi027_stage_6")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_026；输入必须包含 trade_time、code、high、openint、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


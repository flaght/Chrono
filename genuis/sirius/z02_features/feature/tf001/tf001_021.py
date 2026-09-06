"""
因子代号: tf001_021
原历史名: tf001_021
因子定义: 持仓量收益率相关性：持仓量变化率与收盘收益率的滚动相关系数
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
DEFAULT_QUANTILE=0.8
NAME="tf001_021"


def calculate(df_lazy: pl.LazyFrame, period: int, quantile: float) -> pl.LazyFrame:
    """使用 trade_time、code、close、openint 计算 tf001_021 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).shift(1).over("code")).alias("_oi022_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_oi022_stage_0")).is_not_null() & ((pl.col("_oi022_stage_0")) != 0)).then((pl.col('close')) / (pl.col("_oi022_stage_0"))).otherwise(None)).alias("_oi022_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_oi022_stage_1")) > 0).then((pl.col("_oi022_stage_1")).log()).otherwise(None)).alias("_oi022_stage_2")
        )
        .with_columns(
            ((pl.col("_oi022_stage_2")) * (pl.col('openint'))).alias("_oi022_stage_3")
        )
        .with_columns(
            ((pl.col('openint')).rolling_sum(period).over("code")).alias("_oi022_stage_4")
        )
        .with_columns(
            (pl.when((pl.col("_oi022_stage_4")).is_not_null() & ((pl.col("_oi022_stage_4")) != 0)).then((pl.col("_oi022_stage_3")) / (pl.col("_oi022_stage_4"))).otherwise(None)).alias("_oi022_stage_5")
        )
        .with_columns(
            ((pl.col("_oi022_stage_5")).rolling_quantile(quantile, window_size=period).over("code")).alias("_oi022_stage_6")
        )
        .with_columns(
            ((pl.col("_oi022_stage_5")) >= (pl.col("_oi022_stage_6"))).alias("_oi022_stage_7")
        )
        .with_columns(
            (pl.when(pl.col("_oi022_stage_7")).then(pl.col("_oi022_stage_5")).otherwise(pl.lit(0.0))).alias("_oi022_stage_8")
        )
        .with_columns(
            ((pl.col("_oi022_stage_8")).rolling_mean(period).over("code")).alias("_oi022_stage_9")
        )
        .with_columns(
            (pl.col("_oi022_stage_9")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD, quantile: float = DEFAULT_QUANTILE) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_021；输入必须包含 trade_time、code、close、openint。"""
    validate_period(period)
    if not 0 < quantile < 1:
        raise ValueError("quantile 必须在 0 和 1 之间")
    return calculate(df_lazy.sort(["trade_time", "code"]), period, quantile).select(["trade_time", "code", NAME])


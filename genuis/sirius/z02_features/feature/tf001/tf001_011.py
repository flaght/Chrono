"""
因子代号: tf001_011
原历史名: tf001_011
因子定义: 持仓占比信息熵：以持仓量占滚动总持仓比重为分布计算的信息熵
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_011"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、openint 计算 tf001_011 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('openint')).rolling_sum(period).over("code")).alias("_oi012_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_oi012_stage_0")).is_not_null() & ((pl.col("_oi012_stage_0")) != 0)).then((pl.col('openint')) / (pl.col("_oi012_stage_0"))).otherwise(None)).alias("_oi012_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_oi012_stage_1")) > 0).then((pl.col("_oi012_stage_1")).log()).otherwise(None)).alias("_oi012_stage_2")
        )
        .with_columns(
            ((pl.col("_oi012_stage_1")) * (pl.col("_oi012_stage_2"))).alias("_oi012_stage_3")
        )
        .with_columns(
            ((pl.col("_oi012_stage_3")).rolling_sum(period).over("code")).alias("_oi012_stage_4")
        )
        .with_columns(
            (pl.col("_oi012_stage_4")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_011；输入必须包含 trade_time、code、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


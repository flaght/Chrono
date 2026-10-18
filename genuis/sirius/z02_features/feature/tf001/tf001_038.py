"""
因子代号: tf001_038
原历史名: tf001_038
因子定义: 持仓量突破强度：持仓量突破 N 周期布林带上轨的幅度指标
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_038"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、close、openint 计算 tf001_038 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).shift(1).over("code")).alias("_oi039_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_oi039_stage_0")).is_not_null() & ((pl.col("_oi039_stage_0")) != 0)).then((pl.col('close')) / (pl.col("_oi039_stage_0"))).otherwise(None)).alias("_oi039_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_oi039_stage_1")) > 0).then((pl.col("_oi039_stage_1")).log()).otherwise(None)).alias("_oi039_stage_2")
        )
        .with_columns(
            ((pl.col("_oi039_stage_2")) * (pl.col('openint'))).alias("_oi039_stage_3")
        )
        .with_columns(
            ((pl.col("_oi039_stage_3")).rolling_sum(period).over("code")).alias("_oi039_stage_4")
        )
        .with_columns(
            ((pl.col('openint')).rolling_sum(period).over("code")).alias("_oi039_stage_5")
        )
        .with_columns(
            (pl.when((pl.col("_oi039_stage_5")).is_not_null() & ((pl.col("_oi039_stage_5")) != 0)).then((pl.col("_oi039_stage_4")) / (pl.col("_oi039_stage_5"))).otherwise(None)).alias("_oi039_stage_6")
        )
        .with_columns(
            (-(pl.col("_oi039_stage_6"))).alias("_oi039_stage_7")
        )
        .with_columns(
            (pl.col("_oi039_stage_7")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_038；输入必须包含 trade_time、code、close、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


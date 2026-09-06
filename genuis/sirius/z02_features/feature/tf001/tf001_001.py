"""
因子代号: tf001_001
原历史名: tf001_001
因子定义: 持仓量符号加权收益率与收益绝对值相关系数及波动比率
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_001"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、close、openint 计算 tf001_001 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).shift(1).over("code")).alias("_oi001_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_oi001_stage_0")).is_not_null() & ((pl.col("_oi001_stage_0")) != 0)).then((pl.col('close')) / (pl.col("_oi001_stage_0"))).otherwise(None)).alias("_oi001_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_oi001_stage_1")) > 0).then((pl.col("_oi001_stage_1")).log()).otherwise(None)).alias("_oi001_stage_2")
        )
        .with_columns(
            ((pl.col("_oi001_stage_2")).abs()).alias("_oi001_stage_3")
        )
        .with_columns(
            (pl.when((pl.col("_oi001_stage_2")) > 0).then(1.0).when((pl.col("_oi001_stage_2")) < 0).then(-1.0).otherwise(0.0)).alias("_oi001_stage_4")
        )
        .with_columns(
            (pl.when((pl.lit(1000000.0)).is_not_null() & ((pl.lit(1000000.0)) != 0)).then((pl.col('openint')) / (pl.lit(1000000.0))).otherwise(None)).alias("_oi001_stage_5")
        )
        .with_columns(
            ((pl.col("_oi001_stage_4")) * (pl.col("_oi001_stage_5"))).alias("_oi001_stage_6")
        )
        .with_columns(
            (pl.rolling_corr(pl.col("_oi001_stage_3"), pl.col("_oi001_stage_6"), window_size=period).over("code")).alias("_oi001_stage_7")
        )
        .with_columns(
            ((pl.col("_oi001_stage_3")).rolling_std(period).over("code")).alias("_oi001_stage_8")
        )
        .with_columns(
            ((pl.col("_oi001_stage_7")) * (pl.col("_oi001_stage_8"))).alias("_oi001_stage_9")
        )
        .with_columns(
            ((pl.col("_oi001_stage_6")).rolling_std(period).over("code")).alias("_oi001_stage_10")
        )
        .with_columns(
            (pl.when((pl.col("_oi001_stage_10")).is_not_null() & ((pl.col("_oi001_stage_10")) != 0)).then((pl.col("_oi001_stage_9")) / (pl.col("_oi001_stage_10"))).otherwise(None)).alias("_oi001_stage_11")
        )
        .with_columns(
            (pl.col("_oi001_stage_11")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_001；输入必须包含 trade_time、code、close、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


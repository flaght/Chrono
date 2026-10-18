"""
因子代号: tf001_039
原历史名: tf001_039
因子定义: 持仓量收敛发散度：短期与长期持仓量移动平均线的相对发散比率
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_039"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、close、openint 计算 tf001_039 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).shift(1).over("code")).alias("_oi040_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_oi040_stage_0")).is_not_null() & ((pl.col("_oi040_stage_0")) != 0)).then((pl.col('close')) / (pl.col("_oi040_stage_0"))).otherwise(None)).alias("_oi040_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_oi040_stage_1")) > 0).then((pl.col("_oi040_stage_1")).log()).otherwise(None)).alias("_oi040_stage_2")
        )
        .with_columns(
            ((pl.col("_oi040_stage_2")).rolling_mean(period).over("code")).alias("_oi040_stage_3")
        )
        .with_columns(
            ((pl.col("_oi040_stage_2")).rolling_std(period).over("code")).alias("_oi040_stage_4")
        )
        .with_columns(
            ((pl.col("_oi040_stage_3")) + (pl.col("_oi040_stage_4"))).alias("_oi040_stage_5")
        )
        .with_columns(
            ((pl.col("_oi040_stage_2")) > (pl.col("_oi040_stage_5"))).alias("_oi040_stage_6")
        )
        .with_columns(
            (pl.when(pl.col("_oi040_stage_6")).then(pl.col('openint')).otherwise(pl.lit(0.0))).alias("_oi040_stage_7")
        )
        .with_columns(
            ((pl.col("_oi040_stage_7")).rolling_std(period).over("code")).alias("_oi040_stage_8")
        )
        .with_columns(
            ((pl.col('openint')).rolling_std(period).over("code")).alias("_oi040_stage_9")
        )
        .with_columns(
            (pl.when((pl.col("_oi040_stage_9")).is_not_null() & ((pl.col("_oi040_stage_9")) != 0)).then((pl.col("_oi040_stage_8")) / (pl.col("_oi040_stage_9"))).otherwise(None)).alias("_oi040_stage_10")
        )
        .with_columns(
            (pl.col("_oi040_stage_10")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_039；输入必须包含 trade_time、code、close、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


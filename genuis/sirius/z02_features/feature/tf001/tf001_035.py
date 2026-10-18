"""
因子代号: tf001_035
原历史名: tf001_035
因子定义: 负持仓量指数 (NVI-OI)：持仓量减少周期内价格变动的累积指数
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
DEFAULT_SCALAR=100.0
NAME="tf001_035"


def calculate(df_lazy: pl.LazyFrame, period: int, scalar: float) -> pl.LazyFrame:
    """使用 trade_time、code、close、openint 计算 tf001_035 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('openint')).diff(1).over("code")).alias("_oi036_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_oi036_stage_0")) > 0).then(1.0).when((pl.col("_oi036_stage_0")) < 0).then(-1.0).otherwise(0.0)).alias("_oi036_stage_1")
        )
        .with_columns(
            ((pl.col("_oi036_stage_1")) < (pl.lit(0))).alias("_oi036_stage_2")
        )
        .with_columns(
            ((pl.col("_oi036_stage_1")).abs()).alias("_oi036_stage_3")
        )
        .with_columns(
            ((pl.col('close')).diff(period).over("code")).alias("_oi036_stage_4")
        )
        .with_columns(
            ((pl.lit(scalar)) * (pl.col("_oi036_stage_4"))).alias("_oi036_stage_5")
        )
        .with_columns(
            ((pl.col('close')).shift(period).over("code")).alias("_oi036_stage_6")
        )
        .with_columns(
            (pl.when((pl.col("_oi036_stage_6")).is_not_null() & ((pl.col("_oi036_stage_6")) != 0)).then((pl.col("_oi036_stage_5")) / (pl.col("_oi036_stage_6"))).otherwise(None)).alias("_oi036_stage_7")
        )
        .with_columns(
            ((pl.col("_oi036_stage_3")) * (pl.col("_oi036_stage_7"))).alias("_oi036_stage_8")
        )
        .with_columns(
            (pl.when(pl.col("_oi036_stage_2")).then(pl.col("_oi036_stage_8")).otherwise(pl.lit(None))).alias("_oi036_stage_9")
        )
        .with_columns(
            ((pl.col("_oi036_stage_9")).rolling_sum(period).over("code")).alias("_oi036_stage_10")
        )
        .with_columns(
            (pl.col("_oi036_stage_10")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD, scalar: float = DEFAULT_SCALAR) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_035；输入必须包含 trade_time、code、close、openint。"""
    validate_period(period)
    if scalar <= 0:
        raise ValueError("scalar 必须为正数")
    return calculate(df_lazy.sort(["trade_time", "code"]), period, scalar).select(["trade_time", "code", NAME])


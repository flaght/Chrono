"""
因子代号: tf001_031
原历史名: tf001_031
因子定义: 持仓量佳庆振荡器 (Chaikin-OI)：持仓 AD 线的短期与长期 EMA 差值
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
DEFAULT_PERIODS=(5,15)
NAME="tf001_031"


def calculate(df_lazy: pl.LazyFrame, periods: tuple[int, int]) -> pl.LazyFrame:
    """使用 trade_time、code、high、low、close、openint 计算 tf001_031 核心公式。"""
    (fast, slow) = periods
    return (
        df_lazy
        .with_columns(
            ((pl.lit(2)) * (pl.col('close'))).alias("_oi032_stage_0")
        )
        .with_columns(
            ((pl.col("_oi032_stage_0")) - (pl.col('high'))).alias("_oi032_stage_1")
        )
        .with_columns(
            ((pl.col("_oi032_stage_1")) - (pl.col('low'))).alias("_oi032_stage_2")
        )
        .with_columns(
            ((pl.col("_oi032_stage_2")) * (pl.col('openint'))).alias("_oi032_stage_3")
        )
        .with_columns(
            ((pl.col('high')) - (pl.col('low'))).alias("_oi032_stage_4")
        )
        .with_columns(
            (pl.when((pl.col("_oi032_stage_4")).is_not_null() & ((pl.col("_oi032_stage_4")) != 0)).then((pl.col("_oi032_stage_3")) / (pl.col("_oi032_stage_4"))).otherwise(None)).alias("_oi032_stage_5")
        )
        .with_columns(
            ((pl.col("_oi032_stage_5")).rolling_mean(fast).over("code")).alias("_oi032_stage_6")
        )
        .with_columns(
            ((pl.col("_oi032_stage_5")).rolling_mean(slow).over("code")).alias("_oi032_stage_7")
        )
        .with_columns(
            ((pl.col("_oi032_stage_6")) - (pl.col("_oi032_stage_7"))).alias("_oi032_stage_8")
        )
        .with_columns(
            (pl.col("_oi032_stage_8")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, periods: tuple[int, int] = DEFAULT_PERIODS) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_031；输入必须包含 trade_time、code、high、low、close、openint。"""
    fast, slow = periods
    validate_period(fast)
    validate_period(slow)
    if fast >= slow:
        raise ValueError("fast 必须小于 slow")
    return calculate(df_lazy.sort(["trade_time", "code"]), periods).select(["trade_time", "code", NAME])


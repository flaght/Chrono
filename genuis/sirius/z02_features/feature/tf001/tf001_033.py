"""
因子代号: tf001_033
原历史名: tf001_033
因子定义: 持仓力指数 (EFI-OI)：价格变化与持仓量乘积的指数移动平均
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_033"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、close、openint 计算 tf001_033 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).diff(1).over("code")).alias("_oi034_stage_0")
        )
        .with_columns(
            ((pl.col("_oi034_stage_0")) * (pl.col('openint'))).alias("_oi034_stage_1")
        )
        .with_columns(
            ((pl.col("_oi034_stage_1")).rolling_mean(period).over("code")).alias("_oi034_stage_2")
        )
        .with_columns(
            (pl.col("_oi034_stage_2")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_033；输入必须包含 trade_time、code、close、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


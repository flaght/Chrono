"""
因子代号: tf001_018
原历史名: tf001_018
因子定义: 累计持仓量均值：累计持仓量在 period 周期内的移动平均
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_018"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、openint 计算 tf001_018 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('openint')).rolling_sum(period).over("code")).alias("_oi019_stage_0")
        )
        .with_columns(
            ((pl.col("_oi019_stage_0")).rolling_mean(period).over("code")).alias("_oi019_stage_1")
        )
        .with_columns(
            (pl.col("_oi019_stage_1")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_018；输入必须包含 trade_time、code、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


"""
因子代号: tf001_012
原历史名: tf001_012
因子定义: 持仓量价协方差：持仓量变化与价格收益率在 period 周期内的滚动协方差
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_012"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、close、openint 计算 tf001_012 核心公式。"""
    return (
        df_lazy
        .with_columns(
            (pl.rolling_corr(pl.col('close'), pl.col('openint'), window_size=period).over("code")).alias("_oi013_stage_0")
        )
        .with_columns(
            (pl.col("_oi013_stage_0")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_012；输入必须包含 trade_time、code、close、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


"""
因子代号: tf001_036
原历史名: tf001_036
因子定义: 能量潮持仓指标 (OBV-OI)：价格涨跌方向符号加权的持仓量累积指标
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_036"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、close、openint 计算 tf001_036 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).diff(1).over("code")).alias("_oi037_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_oi037_stage_0")) > 0).then(1.0).when((pl.col("_oi037_stage_0")) < 0).then(-1.0).otherwise(0.0)).alias("_oi037_stage_1")
        )
        .with_columns(
            ((pl.col("_oi037_stage_1")) * (pl.col('openint'))).alias("_oi037_stage_2")
        )
        .with_columns(
            ((pl.col("_oi037_stage_2")).rolling_sum(period).over("code")).alias("_oi037_stage_3")
        )
        .with_columns(
            (pl.col("_oi037_stage_3")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_036；输入必须包含 trade_time、code、close、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


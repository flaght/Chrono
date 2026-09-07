"""
因子代号: tf001_014
原历史名: tf001_014
因子定义: 高持仓区间价格偏离：最高20%持仓量区间的加权价格相对均价的偏离度
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_014"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、close、openint 计算 tf001_014 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('openint')).rolling_quantile(0.8, window_size=period).over("code")).alias("_oi015_stage_0")
        )
        .with_columns(
            ((pl.col('openint')) >= (pl.col("_oi015_stage_0"))).alias("_oi015_stage_1")
        )
        .with_columns(
            ((pl.col('close')).rolling_mean(period).over("code")).alias("_oi015_stage_2")
        )
        .with_columns(
            ((pl.col('close')) - (pl.col("_oi015_stage_2"))).alias("_oi015_stage_3")
        )
        .with_columns(
            (pl.when(pl.col("_oi015_stage_1")).then(pl.col("_oi015_stage_3")).otherwise(pl.lit(None))).alias("_oi015_stage_4")
        )
        .with_columns(
            ((pl.col("_oi015_stage_4")).rolling_mean(period).over("code")).alias("_oi015_stage_5")
        )
        .with_columns(
            (pl.col("_oi015_stage_5")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_014；输入必须包含 trade_time、code、close、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


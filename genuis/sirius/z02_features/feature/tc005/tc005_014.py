"""
因子代号: tc005_014
原历史名: tc005_014
因子定义: N期最高价与最低价极差与成交量变化率的协方差复合因子，衡量极端波动与量能变化的联动性
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
NAME = "tc005_014"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用trade_time、code、high、low、volume计算 tc005_014 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('high')).rolling_max(period).over("code")).alias("_cr020_stage_0")
        )
        .with_columns(
            ((pl.col('low')).rolling_min(period).over("code")).alias("_cr020_stage_1")
        )
        .with_columns(
            ((pl.col("_cr020_stage_0")) - (pl.col("_cr020_stage_1"))).alias("_cr020_stage_2")
        )
        .with_columns(
            ((pl.col('volume')).pct_change().over("code")).alias("_cr020_stage_3")
        )
        .with_columns(
            (pl.rolling_cov(pl.col("_cr020_stage_2"), pl.col("_cr020_stage_3"), window_size=period).over("code")).alias("_cr020_stage_4")
        )
        .with_columns(
            (pl.col("_cr020_stage_4")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc005_014；输入必须包含trade_time、code、high、low、volume。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


"""
因子代号: tc005_020
原历史名: tc005_020
因子定义: N期最高价与最低价极差的自相关系数与收盘价波动率的协方差复合因子，衡量极端波动惯性与风险的联动性
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
NAME = "tc005_020"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用trade_time、code、high、low、close计算 tc005_020 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('high')).rolling_max(period).over("code")).alias("_cr026_stage_0")
        )
        .with_columns(
            ((pl.col('low')).rolling_min(period).over("code")).alias("_cr026_stage_1")
        )
        .with_columns(
            ((pl.col("_cr026_stage_0")) - (pl.col("_cr026_stage_1"))).alias("_cr026_stage_2")
        )
        .with_columns(
            ((pl.col("_cr026_stage_2")).shift(1).over("code")).alias("_cr026_stage_3")
        )
        .with_columns(
            (pl.rolling_corr(pl.col("_cr026_stage_2"), pl.col("_cr026_stage_3"), window_size=period).over("code")).alias("_cr026_stage_4")
        )
        .with_columns(
            ((pl.col('close')).rolling_std(period).over("code")).alias("_cr026_stage_5")
        )
        .with_columns(
            (pl.rolling_cov(pl.col("_cr026_stage_4"), pl.col("_cr026_stage_5"), window_size=period).over("code")).alias("_cr026_stage_6")
        )
        .with_columns(
            (pl.col("_cr026_stage_6")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc005_020；输入必须包含trade_time、code、high、low、close。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


"""
因子代号: tc005_007
原历史名: tc005_007
因子定义: N期最高价与最低价的极差与收盘价波动率复合因子，衡量极端波动与风险
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
NAME = "tc005_007"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用trade_time、code、high、low、close计算 tc005_007 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('high')).rolling_max(period).over("code")).alias("_cr012_stage_0")
        )
        .with_columns(
            ((pl.col('low')).rolling_min(period).over("code")).alias("_cr012_stage_1")
        )
        .with_columns(
            ((pl.col("_cr012_stage_0")) - (pl.col("_cr012_stage_1"))).alias("_cr012_stage_2")
        )
        .with_columns(
            ((pl.col('close')).rolling_std(period).over("code")).alias("_cr012_stage_3")
        )
        .with_columns(
            ((pl.col("_cr012_stage_2")) * (pl.col("_cr012_stage_3"))).alias("_cr012_stage_4")
        )
        .with_columns(
            (pl.col("_cr012_stage_4")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc005_007；输入必须包含trade_time、code、high、low、close。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


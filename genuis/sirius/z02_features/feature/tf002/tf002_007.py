"""
因子代号: tf002_007
原历史名: tf002_007
因子定义: 短期与长期持仓量波动率比值的sigmoid变换因子，衡量持仓波动率聚集与极端变化
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIODS = (5, 15)
NAME = "tf002_007"


def calculate(df_lazy: pl.LazyFrame, periods: tuple[int, int]) -> pl.LazyFrame:
    """使用trade_time、code、openint计算 tf002_007 核心公式。"""
    (fast, slow) = periods
    return (
        df_lazy
        .with_columns(
            ((pl.col('openint')).rolling_std(fast).over("code")).alias("_cr053_stage_0")
        )
        .with_columns(
            ((pl.col('openint')).rolling_std(slow).over("code")).alias("_cr053_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_cr053_stage_1")).is_not_null() & ((pl.col("_cr053_stage_1")) != 0)).then((pl.col("_cr053_stage_0")) / (pl.col("_cr053_stage_1"))).otherwise(None)).alias("_cr053_stage_2")
        )
        .with_columns(
            (-(pl.col("_cr053_stage_2"))).alias("_cr053_stage_3")
        )
        .with_columns(
            ((pl.col("_cr053_stage_3")).exp()).alias("_cr053_stage_4")
        )
        .with_columns(
            ((pl.lit(1)) + (pl.col("_cr053_stage_4"))).alias("_cr053_stage_5")
        )
        .with_columns(
            (pl.when((pl.col("_cr053_stage_5")).is_not_null() & ((pl.col("_cr053_stage_5")) != 0)).then((pl.lit(1)) / (pl.col("_cr053_stage_5"))).otherwise(None)).alias("_cr053_stage_6")
        )
        .with_columns(
            (pl.col("_cr053_stage_6")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, periods: tuple[int, int] = DEFAULT_PERIODS) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf002_007；输入必须包含trade_time、code、openint。"""
    if len(periods) != 2:
        raise ValueError("periods 必须包含 fast、slow 两个周期")
    fast, slow = periods
    validate_period(fast)
    validate_period(slow)
    if fast >= slow:
        raise ValueError("fast 必须小于 slow")
    return calculate(df_lazy.sort(["trade_time", "code"]), periods).select(["trade_time", "code", NAME])


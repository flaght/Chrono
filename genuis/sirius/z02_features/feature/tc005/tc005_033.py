"""
因子代号: tc005_033
原历史名: tc005_033
因子定义: 短期与长期波动率比值的sigmoid变换因子，衡量波动率聚集与极端变化
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIODS = (5, 15)
NAME = "tc005_033"


def calculate(df_lazy: pl.LazyFrame, periods: tuple[int, int]) -> pl.LazyFrame:
    """使用trade_time、code、close计算 tc005_033 核心公式。"""
    (fast, slow) = periods
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).shift(1).over("code")).alias("_cr041_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_cr041_stage_0")).is_not_null() & ((pl.col("_cr041_stage_0")) != 0)).then((pl.col('close')) / (pl.col("_cr041_stage_0"))).otherwise(None)).alias("_cr041_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_cr041_stage_1")) > 0).then((pl.col("_cr041_stage_1")).log()).otherwise(None)).alias("_cr041_stage_2")
        )
        .with_columns(
            ((pl.col("_cr041_stage_2")).rolling_std(fast).over("code")).alias("_cr041_stage_3")
        )
        .with_columns(
            ((pl.col("_cr041_stage_2")).rolling_std(slow).over("code")).alias("_cr041_stage_4")
        )
        .with_columns(
            (pl.when((pl.col("_cr041_stage_4")).is_not_null() & ((pl.col("_cr041_stage_4")) != 0)).then((pl.col("_cr041_stage_3")) / (pl.col("_cr041_stage_4"))).otherwise(None)).alias("_cr041_stage_5")
        )
        .with_columns(
            (-(pl.col("_cr041_stage_5"))).alias("_cr041_stage_6")
        )
        .with_columns(
            ((pl.col("_cr041_stage_6")).exp()).alias("_cr041_stage_7")
        )
        .with_columns(
            ((pl.lit(1)) + (pl.col("_cr041_stage_7"))).alias("_cr041_stage_8")
        )
        .with_columns(
            (pl.when((pl.col("_cr041_stage_8")).is_not_null() & ((pl.col("_cr041_stage_8")) != 0)).then((pl.lit(1)) / (pl.col("_cr041_stage_8"))).otherwise(None)).alias("_cr041_stage_9")
        )
        .with_columns(
            (pl.col("_cr041_stage_9")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, periods: tuple[int, int] = DEFAULT_PERIODS) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc005_033；输入必须包含trade_time、code、close。"""
    if len(periods) != 2:
        raise ValueError("periods 必须包含 fast、slow 两个周期")
    fast, slow = periods
    validate_period(fast)
    validate_period(slow)
    if fast >= slow:
        raise ValueError("fast 必须小于 slow")
    return calculate(df_lazy.sort(["trade_time", "code"]), periods).select(["trade_time", "code", NAME])


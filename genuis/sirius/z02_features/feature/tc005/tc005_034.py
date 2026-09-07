"""
因子代号: tc005_034
原历史名: tc005_034
因子定义: 收盘价与长期均值偏离度的tanh非线性压缩因子，衡量均值回归强度
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
NAME = "tc005_034"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用trade_time、code、close计算 tc005_034 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).rolling_mean(period).over("code")).alias("_cr042_stage_0")
        )
        .with_columns(
            ((pl.col('close')) - (pl.col("_cr042_stage_0"))).alias("_cr042_stage_1")
        )
        .with_columns(
            ((pl.col('close')).rolling_std(period).over("code")).alias("_cr042_stage_2")
        )
        .with_columns(
            (pl.when((pl.col("_cr042_stage_2")).is_not_null() & ((pl.col("_cr042_stage_2")) != 0)).then((pl.col("_cr042_stage_1")) / (pl.col("_cr042_stage_2"))).otherwise(None)).alias("_cr042_stage_3")
        )
        .with_columns(
            ((pl.col("_cr042_stage_3")).tanh()).alias("_cr042_stage_4")
        )
        .with_columns(
            (pl.col("_cr042_stage_4")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc005_034；输入必须包含trade_time、code、close。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


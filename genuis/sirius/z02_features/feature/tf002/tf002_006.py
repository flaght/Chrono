"""
因子代号: tf002_006
原历史名: tf002_006
因子定义: 持仓量与长期均值偏离度的tanh非线性压缩因子，衡量持仓均值回归强度
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
NAME = "tf002_006"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用trade_time、code、openint计算 tf002_006 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('openint')).rolling_mean(period).over("code")).alias("_cr052_stage_0")
        )
        .with_columns(
            ((pl.col('openint')) - (pl.col("_cr052_stage_0"))).alias("_cr052_stage_1")
        )
        .with_columns(
            ((pl.col('openint')).rolling_std(period).over("code")).alias("_cr052_stage_2")
        )
        .with_columns(
            (pl.when((pl.col("_cr052_stage_2")).is_not_null() & ((pl.col("_cr052_stage_2")) != 0)).then((pl.col("_cr052_stage_1")) / (pl.col("_cr052_stage_2"))).otherwise(None)).alias("_cr052_stage_3")
        )
        .with_columns(
            ((pl.col("_cr052_stage_3")).tanh()).alias("_cr052_stage_4")
        )
        .with_columns(
            (pl.col("_cr052_stage_4")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf002_006；输入必须包含trade_time、code、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


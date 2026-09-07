"""
因子代号: tf001_037
原历史名: tf001_037
因子定义: Alpha 152 衍生持仓因子：价格收益率与持仓量排名的滚动协方差与极值组合
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_037"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、value、openint、close 计算 tf001_037 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('value')).rolling_sum(period).over("code")).alias("_oi038_stage_0")
        )
        .with_columns(
            ((pl.col("_oi038_stage_0")).rolling_mean(period).over("code")).alias("_oi038_stage_1")
        )
        .with_columns(
            ((pl.col('openint')).rolling_sum(period).over("code")).alias("_oi038_stage_2")
        )
        .with_columns(
            ((pl.col("_oi038_stage_2")).rolling_mean(period).over("code")).alias("_oi038_stage_3")
        )
        .with_columns(
            (pl.when((pl.col("_oi038_stage_3")).is_not_null() & ((pl.col("_oi038_stage_3")) != 0)).then((pl.col("_oi038_stage_1")) / (pl.col("_oi038_stage_3"))).otherwise(None)).alias("_oi038_stage_4")
        )
        .with_columns(
            (pl.when((pl.col("_oi038_stage_4")).is_not_null() & ((pl.col("_oi038_stage_4")) != 0)).then((pl.col('close')) / (pl.col("_oi038_stage_4"))).otherwise(None)).alias("_oi038_stage_5")
        )
        .with_columns(
            (pl.col("_oi038_stage_5")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_037；输入必须包含 trade_time、code、value、openint、close。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


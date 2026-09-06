"""
因子代号: tf001_029
原历史名: tf001_029
因子定义: 持仓量分位数位置：持仓量在 period 周期历史区间内的百分比排位
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_029"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、openint 计算 tf001_029 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('openint')) > (pl.lit(0))).alias("_oi030_stage_0")
        )
        .with_columns(
            (pl.when(pl.col("_oi030_stage_0")).then(pl.col('openint')).otherwise(pl.lit(None))).alias("_oi030_stage_1")
        )
        .with_columns(
            ((pl.col("_oi030_stage_1")).shift(1).over("code")).alias("_oi030_stage_2")
        )
        .with_columns(
            (pl.when((pl.col("_oi030_stage_2")).is_not_null() & ((pl.col("_oi030_stage_2")) != 0)).then((pl.col("_oi030_stage_1")) / (pl.col("_oi030_stage_2"))).otherwise(None)).alias("_oi030_stage_3")
        )
        .with_columns(
            (pl.when((pl.col("_oi030_stage_3")) > 0).then((pl.col("_oi030_stage_3")).log()).otherwise(None)).alias("_oi030_stage_4")
        )
        .with_columns(
            ((pl.col("_oi030_stage_4")).rolling_std(period).over("code")).alias("_oi030_stage_5")
        )
        .with_columns(
            (pl.col("_oi030_stage_5")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_029；输入必须包含 trade_time、code、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


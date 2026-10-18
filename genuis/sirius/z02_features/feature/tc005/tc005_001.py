"""
因子代号: tc005_001
原历史名: tc005_001
因子定义: N期最高价与最低价的对数比率波动因子，衡量价格区间的波动幅度
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
NAME = "tc005_001"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用trade_time、code、high、low计算 tc005_001 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('high')).rolling_max(period).over("code")).alias("_cr003_stage_0")
        )
        .with_columns(
            ((pl.col('low')).rolling_min(period).over("code")).alias("_cr003_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_cr003_stage_1")).is_not_null() & ((pl.col("_cr003_stage_1")) != 0)).then((pl.col("_cr003_stage_0")) / (pl.col("_cr003_stage_1"))).otherwise(None)).alias("_cr003_stage_2")
        )
        .with_columns(
            (pl.when((pl.col("_cr003_stage_2")) > 0).then((pl.col("_cr003_stage_2")).log()).otherwise(None)).alias("_cr003_stage_3")
        )
        .with_columns(
            (pl.col("_cr003_stage_3")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc005_001；输入必须包含trade_time、code、high、low。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


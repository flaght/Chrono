"""
因子代号: tc005_002
原历史名: tc005_002
因子定义: N期最高价与收盘价的相关系数因子，衡量价格高点与收盘的同步性
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
NAME = "tc005_002"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用trade_time、code、high、close计算 tc005_002 核心公式。"""
    return (
        df_lazy
        .with_columns(
            (pl.rolling_corr(pl.col('high'), pl.col('close'), window_size=period).over("code")).alias("_cr006_stage_0")
        )
        .with_columns(
            (pl.col("_cr006_stage_0")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc005_002；输入必须包含trade_time、code、high、close。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


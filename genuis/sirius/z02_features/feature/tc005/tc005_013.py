"""
因子代号: tc005_013
原历史名: tc005_013
因子定义: N期收盘价与开盘价的对数收益率与成交量波动率的相关系数复合因子，衡量收益与量能风险的同步性
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
NAME = "tc005_013"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用trade_time、code、close、open、volume计算 tc005_013 核心公式。"""
    return (
        df_lazy
        .with_columns(
            (pl.when((pl.col('open')).is_not_null() & ((pl.col('open')) != 0)).then((pl.col('close')) / (pl.col('open'))).otherwise(None)).alias("_cr019_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_cr019_stage_0")) > 0).then((pl.col("_cr019_stage_0")).log()).otherwise(None)).alias("_cr019_stage_1")
        )
        .with_columns(
            ((pl.col('volume')).rolling_std(period).over("code")).alias("_cr019_stage_2")
        )
        .with_columns(
            (pl.rolling_corr(pl.col("_cr019_stage_1"), pl.col("_cr019_stage_2"), window_size=period).over("code")).alias("_cr019_stage_3")
        )
        .with_columns(
            (pl.col("_cr019_stage_3")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc005_013；输入必须包含trade_time、code、close、open、volume。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


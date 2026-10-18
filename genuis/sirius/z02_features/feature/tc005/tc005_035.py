"""
因子代号: tc005_035
原历史名: tc005_035
因子定义: 高低价极差的自适应分位阈值触发因子，衡量极端行情爆发概率
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
DEFAULT_QUANTILE = 0.9
NAME = "tc005_035"


def calculate(df_lazy: pl.LazyFrame, period: int, quantile: float) -> pl.LazyFrame:
    """使用trade_time、code、high、low计算 tc005_035 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('high')).rolling_max(period).over("code")).alias("_cr044_stage_0")
        )
        .with_columns(
            ((pl.col('low')).rolling_min(period).over("code")).alias("_cr044_stage_1")
        )
        .with_columns(
            ((pl.col("_cr044_stage_0")) - (pl.col("_cr044_stage_1"))).alias("_cr044_stage_2")
        )
        .with_columns(
            ((pl.col("_cr044_stage_2")).rolling_quantile(quantile, window_size=period).over("code")).alias("_cr044_stage_3")
        )
        .with_columns(
            ((pl.col("_cr044_stage_2")) > (pl.col("_cr044_stage_3"))).alias("_cr044_stage_4")
        )
        .with_columns(
            (pl.when(pl.col("_cr044_stage_4")).then(pl.lit(1.0)).otherwise(pl.lit(0.0))).alias("_cr044_stage_5")
        )
        .with_columns(
            (pl.col("_cr044_stage_5")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD, quantile: float = DEFAULT_QUANTILE) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc005_035；输入必须包含trade_time、code、high、low。"""
    validate_period(period)
    if not 0 < quantile < 1:
        raise ValueError("quantile 必须位于 0 与 1 之间")
    return calculate(df_lazy.sort(["trade_time", "code"]), period, quantile).select(["trade_time", "code", NAME])


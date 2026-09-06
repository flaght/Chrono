"""
因子代号: tf002_014
原历史名: tf002_014
因子定义: N期收盘价对数收益率、最高价极差、持仓量变化率三者的三阶混合分位数复合因子，衡量收益、极端波动与持仓变化的高阶极端分布
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
DEFAULT_QUANTILE = 0.9
NAME = "tf002_014"


def calculate(df_lazy: pl.LazyFrame, period: int, quantile: float) -> pl.LazyFrame:
    """使用trade_time、code、close、high、low、openint计算 tf002_014 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).shift(1).over("code")).alias("_cr063_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_cr063_stage_0")).is_not_null() & ((pl.col("_cr063_stage_0")) != 0)).then((pl.col('close')) / (pl.col("_cr063_stage_0"))).otherwise(None)).alias("_cr063_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_cr063_stage_1")) > 0).then((pl.col("_cr063_stage_1")).log()).otherwise(None)).alias("_cr063_stage_2")
        )
        .with_columns(
            ((pl.col("_cr063_stage_2")).rolling_mean(period).over("code")).alias("_cr063_stage_3")
        )
        .with_columns(
            ((pl.col("_cr063_stage_2")) - (pl.col("_cr063_stage_3"))).alias("_cr063_stage_4")
        )
        .with_columns(
            ((pl.col('high')).rolling_max(period).over("code")).alias("_cr063_stage_5")
        )
        .with_columns(
            ((pl.col('low')).rolling_min(period).over("code")).alias("_cr063_stage_6")
        )
        .with_columns(
            ((pl.col("_cr063_stage_5")) - (pl.col("_cr063_stage_6"))).alias("_cr063_stage_7")
        )
        .with_columns(
            ((pl.col("_cr063_stage_7")).rolling_mean(period).over("code")).alias("_cr063_stage_8")
        )
        .with_columns(
            ((pl.col("_cr063_stage_7")) - (pl.col("_cr063_stage_8"))).alias("_cr063_stage_9")
        )
        .with_columns(
            ((pl.col("_cr063_stage_4")) * (pl.col("_cr063_stage_9"))).alias("_cr063_stage_10")
        )
        .with_columns(
            ((pl.col('openint')).pct_change().over("code")).alias("_cr063_stage_11")
        )
        .with_columns(
            ((pl.col("_cr063_stage_11")).rolling_mean(period).over("code")).alias("_cr063_stage_12")
        )
        .with_columns(
            ((pl.col("_cr063_stage_11")) - (pl.col("_cr063_stage_12"))).alias("_cr063_stage_13")
        )
        .with_columns(
            ((pl.col("_cr063_stage_10")) * (pl.col("_cr063_stage_13"))).alias("_cr063_stage_14")
        )
        .with_columns(
            ((pl.col("_cr063_stage_14")).rolling_quantile(quantile, window_size=period).over("code")).alias("_cr063_stage_15")
        )
        .with_columns(
            (pl.col("_cr063_stage_15")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD, quantile: float = DEFAULT_QUANTILE) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf002_014；输入必须包含trade_time、code、close、high、low、openint。"""
    validate_period(period)
    if not 0 < quantile < 1:
        raise ValueError("quantile 必须位于 0 与 1 之间")
    return calculate(df_lazy.sort(["trade_time", "code"]), period, quantile).select(["trade_time", "code", NAME])


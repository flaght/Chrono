"""
因子代号: tf002_012
原历史名: tf002_012
因子定义: N期收盘价对数收益率、最高价极差、持仓量变化率三者的三阶混合极端值复合因子，衡量收益、极端波动与持仓变化的高阶极端风险
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
NAME = "tf002_012"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用trade_time、code、close、high、low、openint计算 tf002_012 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).shift(1).over("code")).alias("_cr060_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_cr060_stage_0")).is_not_null() & ((pl.col("_cr060_stage_0")) != 0)).then((pl.col('close')) / (pl.col("_cr060_stage_0"))).otherwise(None)).alias("_cr060_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_cr060_stage_1")) > 0).then((pl.col("_cr060_stage_1")).log()).otherwise(None)).alias("_cr060_stage_2")
        )
        .with_columns(
            ((pl.col("_cr060_stage_2")).rolling_mean(period).over("code")).alias("_cr060_stage_3")
        )
        .with_columns(
            ((pl.col("_cr060_stage_2")) - (pl.col("_cr060_stage_3"))).alias("_cr060_stage_4")
        )
        .with_columns(
            ((pl.col('high')).rolling_max(period).over("code")).alias("_cr060_stage_5")
        )
        .with_columns(
            ((pl.col('low')).rolling_min(period).over("code")).alias("_cr060_stage_6")
        )
        .with_columns(
            ((pl.col("_cr060_stage_5")) - (pl.col("_cr060_stage_6"))).alias("_cr060_stage_7")
        )
        .with_columns(
            ((pl.col("_cr060_stage_7")).rolling_mean(period).over("code")).alias("_cr060_stage_8")
        )
        .with_columns(
            ((pl.col("_cr060_stage_7")) - (pl.col("_cr060_stage_8"))).alias("_cr060_stage_9")
        )
        .with_columns(
            ((pl.col("_cr060_stage_4")) * (pl.col("_cr060_stage_9"))).alias("_cr060_stage_10")
        )
        .with_columns(
            ((pl.col('openint')).pct_change().over("code")).alias("_cr060_stage_11")
        )
        .with_columns(
            ((pl.col("_cr060_stage_11")).rolling_mean(period).over("code")).alias("_cr060_stage_12")
        )
        .with_columns(
            ((pl.col("_cr060_stage_11")) - (pl.col("_cr060_stage_12"))).alias("_cr060_stage_13")
        )
        .with_columns(
            ((pl.col("_cr060_stage_10")) * (pl.col("_cr060_stage_13"))).alias("_cr060_stage_14")
        )
        .with_columns(
            ((pl.col("_cr060_stage_14")).rolling_max(period).over("code")).alias("_cr060_stage_15")
        )
        .with_columns(
            ((pl.col("_cr060_stage_14")).rolling_min(period).over("code")).alias("_cr060_stage_16")
        )
        .with_columns(
            ((pl.col("_cr060_stage_15")) - (pl.col("_cr060_stage_16"))).alias("_cr060_stage_17")
        )
        .with_columns(
            (pl.col("_cr060_stage_17")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf002_012；输入必须包含trade_time、code、close、high、low、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


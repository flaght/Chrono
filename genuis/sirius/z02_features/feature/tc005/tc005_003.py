"""
因子代号: tc005_003
原历史名: tc005_003
因子定义: N期收盘价动量与波动率复合因子，衡量价格趋势与风险的综合效应
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
NAME = "tc005_003"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用trade_time、code、close计算 tc005_003 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).shift(period).over("code")).alias("_cr007_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_cr007_stage_0")).is_not_null() & ((pl.col("_cr007_stage_0")) != 0)).then((pl.col('close')) / (pl.col("_cr007_stage_0"))).otherwise(None)).alias("_cr007_stage_1")
        )
        .with_columns(
            ((pl.col("_cr007_stage_1")) - (pl.lit(1))).alias("_cr007_stage_2")
        )
        .with_columns(
            ((pl.col("_cr007_stage_2")).rolling_mean(period).over("code")).alias("_cr007_stage_3")
        )
        .with_columns(
            ((pl.col("_cr007_stage_2")) - (pl.col("_cr007_stage_3"))).alias("_cr007_stage_4")
        )
        .with_columns(
            ((pl.col("_cr007_stage_2")).rolling_std(period).over("code")).alias("_cr007_stage_5")
        )
        .with_columns(
            (pl.when((pl.col("_cr007_stage_5")).is_not_null() & ((pl.col("_cr007_stage_5")) != 0)).then((pl.col("_cr007_stage_4")) / (pl.col("_cr007_stage_5"))).otherwise(None)).alias("_cr007_stage_6")
        )
        .with_columns(
            ((pl.col('close')).shift(1).over("code")).alias("_cr007_stage_7")
        )
        .with_columns(
            (pl.when((pl.col("_cr007_stage_7")).is_not_null() & ((pl.col("_cr007_stage_7")) != 0)).then((pl.col('close')) / (pl.col("_cr007_stage_7"))).otherwise(None)).alias("_cr007_stage_8")
        )
        .with_columns(
            (pl.when((pl.col("_cr007_stage_8")) > 0).then((pl.col("_cr007_stage_8")).log()).otherwise(None)).alias("_cr007_stage_9")
        )
        .with_columns(
            ((pl.col("_cr007_stage_9")).rolling_std(period).over("code")).alias("_cr007_stage_10")
        )
        .with_columns(
            ((pl.col("_cr007_stage_10")).rolling_mean(period).over("code")).alias("_cr007_stage_11")
        )
        .with_columns(
            ((pl.col("_cr007_stage_10")) - (pl.col("_cr007_stage_11"))).alias("_cr007_stage_12")
        )
        .with_columns(
            ((pl.col("_cr007_stage_10")).rolling_std(period).over("code")).alias("_cr007_stage_13")
        )
        .with_columns(
            (pl.when((pl.col("_cr007_stage_13")).is_not_null() & ((pl.col("_cr007_stage_13")) != 0)).then((pl.col("_cr007_stage_12")) / (pl.col("_cr007_stage_13"))).otherwise(None)).alias("_cr007_stage_14")
        )
        .with_columns(
            ((pl.col("_cr007_stage_6")) * (pl.col("_cr007_stage_14"))).alias("_cr007_stage_15")
        )
        .with_columns(
            (pl.col("_cr007_stage_15")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc005_003；输入必须包含trade_time、code、close。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


"""
因子代号: tf001_032
原历史名: tf001_032
因子定义: 持仓资金流量指标 (CMF-OI)：以持仓量加权的资金流量在窗口内的比率
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_032"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、high、low、close、openint 计算 tf001_032 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.lit(2)) * (pl.col('close'))).alias("_oi033_stage_0")
        )
        .with_columns(
            ((pl.col("_oi033_stage_0")) - (pl.col('high'))).alias("_oi033_stage_1")
        )
        .with_columns(
            ((pl.col("_oi033_stage_1")) - (pl.col('low'))).alias("_oi033_stage_2")
        )
        .with_columns(
            ((pl.col("_oi033_stage_2")) * (pl.col('openint'))).alias("_oi033_stage_3")
        )
        .with_columns(
            ((pl.col('high')) - (pl.col('low'))).alias("_oi033_stage_4")
        )
        .with_columns(
            (pl.when((pl.col("_oi033_stage_4")).is_not_null() & ((pl.col("_oi033_stage_4")) != 0)).then((pl.col("_oi033_stage_3")) / (pl.col("_oi033_stage_4"))).otherwise(None)).alias("_oi033_stage_5")
        )
        .with_columns(
            ((pl.col("_oi033_stage_5")).rolling_mean(period).over("code")).alias("_oi033_stage_6")
        )
        .with_columns(
            ((pl.col('openint')).rolling_mean(period).over("code")).alias("_oi033_stage_7")
        )
        .with_columns(
            (pl.when((pl.col("_oi033_stage_7")).is_not_null() & ((pl.col("_oi033_stage_7")) != 0)).then((pl.col("_oi033_stage_6")) / (pl.col("_oi033_stage_7"))).otherwise(None)).alias("_oi033_stage_8")
        )
        .with_columns(
            (pl.col("_oi033_stage_8")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_032；输入必须包含 trade_time、code、high、low、close、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


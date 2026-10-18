"""
因子代号: tf001_027
原历史名: tf001_027
因子定义: 持仓量加权典型价格：以持仓量为权重的典型价格移动均线偏离度
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD=15
NAME="tf001_027"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、open、high、openint 计算 tf001_027 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('openint')).arctan()).alias("_oi028_stage_0")
        )
        .with_columns(
            (pl.rolling_corr(pl.col('openint'), pl.col('high'), window_size=period).over("code")).alias("_oi028_stage_1")
        )
        .with_columns(
            ((pl.col("_oi028_stage_0")) * (pl.col("_oi028_stage_1"))).alias("_oi028_stage_2")
        )
        .with_columns(
            ((pl.col("_oi028_stage_2")) > (pl.lit(0))).alias("_oi028_stage_3")
        )
        .with_columns(
            (pl.when(pl.col("_oi028_stage_3")).then(pl.col("_oi028_stage_2")).otherwise(pl.lit(None))).alias("_oi028_stage_4")
        )
        .with_columns(
            (pl.when((pl.col("_oi028_stage_4")) > 0).then((pl.col("_oi028_stage_4")).log()).otherwise(None)).alias("_oi028_stage_5")
        )
        .with_columns(
            ((pl.col('open')).diff(6).over("code")).alias("_oi028_stage_6")
        )
        .with_columns(
            (pl.min_horizontal(pl.col("_oi028_stage_5"), pl.col("_oi028_stage_6"))).alias("_oi028_stage_7")
        )
        .with_columns(
            (pl.col("_oi028_stage_7")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_027；输入必须包含 trade_time、code、open、high、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


"""
因子代号: tc005_005
原历史名: tc005_005
因子定义: N日收盘价与成交量的相关性与极端值复合因子，衡量价量共振与极端波动
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
NAME = "tc005_005"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用trade_time、code、close、volume计算 tc005_005 核心公式。"""
    return (
        df_lazy
        .with_columns(
            (pl.rolling_corr(pl.col('close'), pl.col('volume'), window_size=period).over("code")).alias("_cr009_stage_0")
        )
        .with_columns(
            ((pl.col('close')).shift(period).over("code")).alias("_cr009_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_cr009_stage_1")).is_not_null() & ((pl.col("_cr009_stage_1")) != 0)).then((pl.col('close')) / (pl.col("_cr009_stage_1"))).otherwise(None)).alias("_cr009_stage_2")
        )
        .with_columns(
            ((pl.col("_cr009_stage_2")) - (pl.lit(1))).alias("_cr009_stage_3")
        )
        .with_columns(
            ((pl.col("_cr009_stage_3")).abs()).alias("_cr009_stage_4")
        )
        .with_columns(
            ((pl.col("_cr009_stage_4")).rolling_quantile(0.95, window_size=period).over("code")).alias("_cr009_stage_5")
        )
        .with_columns(
            ((pl.col("_cr009_stage_4")) > (pl.col("_cr009_stage_5"))).alias("_cr009_stage_6")
        )
        .with_columns(
            (pl.when(pl.col("_cr009_stage_6")).then(pl.lit(1.0)).otherwise(pl.lit(0.0))).alias("_cr009_stage_7")
        )
        .with_columns(
            ((pl.col("_cr009_stage_0")) * (pl.col("_cr009_stage_7"))).alias("_cr009_stage_8")
        )
        .with_columns(
            (pl.col("_cr009_stage_8")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc005_005；输入必须包含trade_time、code、close、volume。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


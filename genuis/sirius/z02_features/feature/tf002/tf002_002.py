"""
因子代号: tf002_002
原历史名: tf002_002
因子定义: N期收盘价与持仓量的相关性与极端值复合因子，衡量价持仓共振与极端波动
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
NAME = "tf002_002"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用trade_time、code、close、openint计算 tf002_002 核心公式。"""
    return (
        df_lazy
        .with_columns(
            (pl.rolling_corr(pl.col('close'), pl.col('openint'), window_size=period).over("code")).alias("_cr047_stage_0")
        )
        .with_columns(
            ((pl.col('close')).shift(period).over("code")).alias("_cr047_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_cr047_stage_1")).is_not_null() & ((pl.col("_cr047_stage_1")) != 0)).then((pl.col('close')) / (pl.col("_cr047_stage_1"))).otherwise(None)).alias("_cr047_stage_2")
        )
        .with_columns(
            ((pl.col("_cr047_stage_2")) - (pl.lit(1))).alias("_cr047_stage_3")
        )
        .with_columns(
            ((pl.col("_cr047_stage_3")).abs()).alias("_cr047_stage_4")
        )
        .with_columns(
            ((pl.col("_cr047_stage_4")).rolling_quantile(0.95, window_size=period).over("code")).alias("_cr047_stage_5")
        )
        .with_columns(
            ((pl.col("_cr047_stage_4")) > (pl.col("_cr047_stage_5"))).alias("_cr047_stage_6")
        )
        .with_columns(
            (pl.when(pl.col("_cr047_stage_6")).then(pl.lit(1.0)).otherwise(pl.lit(0.0))).alias("_cr047_stage_7")
        )
        .with_columns(
            ((pl.col("_cr047_stage_0")) * (pl.col("_cr047_stage_7"))).alias("_cr047_stage_8")
        )
        .with_columns(
            (pl.col("_cr047_stage_8")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf002_002；输入必须包含trade_time、code、close、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])


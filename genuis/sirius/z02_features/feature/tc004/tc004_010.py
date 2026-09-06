"""
因子代号: tc004_010
原历史名: tc004_010
因子定义: 成交额加权下影线强度：以成交额为权重的下影线支撑力度
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD = 15
NAME = "tc004_010"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    return (
        df_lazy
        .with_columns(
            log_return().abs().alias("_r"),
            safe_div(
                pl.col("value") - pl.col("value").rolling_mean(period).over("code"),
                pl.col("value").rolling_std(period).over("code"),
            ).alias("_z"),
        )
        .with_columns(
            pl.col("_z").shift(1).over("code").alias("_previous_z")
        )
        .with_columns(
            pl.rolling_corr(
                pl.col("_r"),
                pl.col("_previous_z"),
                window_size=period,
            )
            .over("code")
            .alias(NAME)
        )
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc004_010；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])

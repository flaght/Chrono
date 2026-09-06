"""
因子代号: tc004_009
原历史名: tc004_009
因子定义: 高低价振幅与成交额弹性：成交额相对价格振幅的边际弹性系数
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD = 15
NAME = "tc004_009"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    return (
        df_lazy
        .with_columns(
            log_return().alias("_r"),
            pl.col("value").shift(1).over("code").alias("_previous_value"),
        )
        .with_columns(
            pl.rolling_corr(
                pl.col("_r"),
                pl.col("_previous_value"),
                window_size=period,
            )
            .over("code")
            .alias(NAME)
        )
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc004_009；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])

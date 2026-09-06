"""
因子代号: tc003_014
原历史名: tc003_014
因子定义: 日内多空力量失衡比：(close - low - (high - close)) / (high - low)
"""
import polars as pl
from feature.utils.common import rolling_rank, safe_div, validate_period
DEFAULT_PERIOD=15
NAME="tc003_014"

def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(safe_div(pl.col("value"),pl.col("volume")).alias("_vwap"),pl.min_horizontal(pl.col("open").rolling_max(period).over("code"),pl.col("close").rolling_max(period).over("code")).alias("_m"))
    return x.with_columns(
        (
            pl.col("_m") * pl.col("low")
            + safe_div(
                rolling_rank(pl.col("_vwap"), period).over("code"),
                pl.col("low"),
            )
        ).alias(NAME)
    )

def compute(df_lazy: pl.LazyFrame, period: int=DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc003_014；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

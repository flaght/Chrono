"""
因子代号: tc003_008
原历史名: tc003_008
因子定义: 价格突破与量能确认：价格创 N 周期新高同时伴随成交量放大的突破信号
"""
import polars as pl
from feature.utils.common import rolling_rank, safe_div, validate_period
DEFAULT_PERIOD=15
NAME="tc003_008"

def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(safe_div(pl.col("value"),pl.col("volume")).alias("_vwap"))
    return x.with_columns(
        (
            pl.col("_vwap").diff(5).over("code")
            - rolling_rank(pl.col("close"), period).over("code").sqrt()
        ).alias(NAME)
    )

def compute(df_lazy: pl.LazyFrame, period: int=DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc003_008；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

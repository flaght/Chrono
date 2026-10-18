"""
因子代号: tc003_003
原历史名: tc003_003
因子定义: VWAP与开盘价偏离相对昨收比率：(rolling_vwap - open) / close.shift(1)
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD=15
NAME="tc003_003"

def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(safe_div(pl.col("value").rolling_sum(period).over("code"),pl.col("volume").rolling_sum(period).over("code")).alias("_avg"))
    return x.with_columns(safe_div(pl.col("_avg")-pl.col("open"),pl.col("close").shift(1).over("code")).alias(NAME))

def compute(df_lazy: pl.LazyFrame, period: int=DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc003_003；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

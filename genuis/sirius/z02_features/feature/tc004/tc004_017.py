"""
因子代号: tc004_017
原历史名: tc004_017
因子定义: 量价方向一致性比率：价格上涨且成交量放大周期数在窗口内的占比
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD = 15
NAME = "tc004_017"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(log_return("volume").alias("_v"),log_return().alias("_c"))
    return x.with_columns(safe_div(pl.rolling_cov(pl.col("_v"),pl.col("_c"),window_size=period).over("code"),pl.col("_c").rolling_var(period).over("code")).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc004_017；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

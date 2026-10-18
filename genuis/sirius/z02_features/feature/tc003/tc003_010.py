"""
因子代号: tc003_010
原历史名: tc003_010
因子定义: 成交量加权高低价极差：以成交量为权重的日内高低价相对极差
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD=15
NAME="tc003_010"

def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(pl.rolling_cov(pl.col("value"),pl.col("high"),window_size=period).over("code").alias("_vh"),pl.col("value").rolling_std(period).over("code").alias("_vs"),pl.rolling_cov(pl.col("volume"),pl.col("value"),window_size=period).over("code").alias("_cov"))
    x=x.with_columns(safe_div(pl.col("_vh"),pl.col("_vs")).alias("_beta"))
    return x.with_columns(safe_div(pl.col("_cov"),pl.col("_beta").rolling_mean(period).over("code")).alias(NAME))

def compute(df_lazy: pl.LazyFrame, period: int=DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc003_010；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

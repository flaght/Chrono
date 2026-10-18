"""
因子代号: tc004_007
原历史名: tc004_007
因子定义: 价格中心距与成交额相关性：价格偏离均值中心距与成交额的协同性
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD = 15
NAME = "tc004_007"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(log_return().alias("_r"))
    x=x.with_columns(pl.col("_r").diff().over("code").alias("_d"))
    x=x.with_columns(pl.col("_d").rolling_mean(period).over("code").alias("_m"),pl.col("_d").rolling_std(period).over("code").alias("_s"))
    return x.with_columns(pl.when(pl.col("_d")>pl.col("_m")+pl.col("_s")).then(pl.col("_r")).otherwise(0.0).rolling_mean(period).over("code").alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc004_007；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])

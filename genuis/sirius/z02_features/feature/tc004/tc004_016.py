"""
因子代号: tc004_016
原历史名: tc004_016
因子定义: 典型价格均线偏离：典型价格 (H+L+C)/3 相对其均线的偏离度
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD = 15
NAME = "tc004_016"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(safe_div(pl.col("close"),pl.col("open")).sub(1).rolling_mean(period).over("code").alias("_sigma"))
    return x.with_columns(safe_div(pl.max_horizontal("open",pl.col("close").shift(1).over("code"))*(1+pl.col("_sigma")),pl.min_horizontal("open",pl.col("close").shift(1).over("code"))*(1-pl.col("_sigma"))).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc004_016；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

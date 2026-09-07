"""
因子代号: tc003_001
原历史名: tc003_001
因子定义: 成交量加权收益率高分位均值：成交量加权收益率超过分位数阈值部分的滚动均值
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD=15
NAME="tc003_001"

def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(log_return().alias("_r"))
    x=x.with_columns(safe_div(pl.col("_r")*pl.col("volume"),pl.col("volume").rolling_sum(period).over("code")).alias("_w"))
    x=x.with_columns(pl.col("_w").rolling_quantile(0.8,window_size=period).over("code").alias("_q"))
    return x.with_columns(pl.when(pl.col("_w")>=pl.col("_q")).then(pl.col("_w")).otherwise(0.0).rolling_mean(period).over("code").alias(NAME))

def compute(df_lazy: pl.LazyFrame, period: int=DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc003_001；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

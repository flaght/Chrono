"""
因子代号: tc003_009
原历史名: tc003_009
因子定义: 实体振幅比的滚动均值：abs(close - open) / (high - low) 的 period 周期移动平均
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD=15
NAME="tc003_009"

def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(pl.col("low").rolling_max(5).over("code").alias("_lm"),pl.col("value").rolling_mean(12).over("code").alias("_vm"))
    x=x.with_columns(pl.rolling_cov(pl.col("_lm"),pl.col("_vm"),window_size=8).over("code").alias("_cov"))
    return x.with_columns(pl.col("_cov").rolling_max(period).over("code").alias(NAME))

def compute(df_lazy: pl.LazyFrame, period: int=DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc003_009；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

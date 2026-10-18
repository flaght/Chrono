"""
因子代号: tc003_006
原历史名: tc003_006
因子定义: 分位数截断收益率动量：收益率在滚动分位数阈值截断后的动量累积
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD=15
NAME="tc003_006"

def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(safe_div(pl.col("value"),pl.col("volume")).alias("_vwap"),pl.col("value").rolling_mean(period).over("code").alias("_mv"))
    x=x.with_columns(pl.col("_vwap").rolling_mean(period).over("code").alias("_vw"))
    x=x.with_columns(pl.col("_vw").rolling_max(period).over("code").alias("_vwm"))
    return x.with_columns(pl.rolling_cov(pl.col("_vwm"),pl.col("_mv"),window_size=period).over("code").alias(NAME))

def compute(df_lazy: pl.LazyFrame, period: int=DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc003_006；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

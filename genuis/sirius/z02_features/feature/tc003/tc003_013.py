"""
因子代号: tc003_013
原历史名: tc003_013
因子定义: 极值价量共振强度：价格极值与成交量极值同时出现的频率与冲击
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD=15
NAME="tc003_013"

def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(safe_div(pl.rolling_cov(pl.col("open"),pl.col("low"),window_size=period).over("code"),pl.col("open").rolling_std(period).over("code")).alias("_c1"))
    x=x.with_columns(safe_div(pl.rolling_cov(pl.col("_c1"),pl.col("close"),window_size=period).over("code"),pl.col("_c1").rolling_std(period).over("code").pow(2)).alias("_c2"))
    return x.with_columns((pl.col("close")-pl.col("_c1")*pl.col("_c2")).alias(NAME))

def compute(df_lazy: pl.LazyFrame, period: int=DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc003_013；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

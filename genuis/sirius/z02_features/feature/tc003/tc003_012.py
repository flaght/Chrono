"""
因子代号: tc003_012
原历史名: tc003_012
因子定义: 价格加速度与量能乘积：价格二阶差分与成交量标准化比值的乘积
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD=15
NAME="tc003_012"

def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(pl.rolling_corr(pl.col("volume"),pl.col("high"),window_size=period).over("code").alias("_corr"))
    x=x.with_columns(pl.when(pl.col("volume").arctan()*pl.col("_corr")>0).then((pl.col("volume").arctan()*pl.col("_corr")).log()).alias("_a"),pl.col("open").diff(6).over("code").alias("_b"))
    return x.with_columns(pl.min_horizontal("_a","_b").alias(NAME))

def compute(df_lazy: pl.LazyFrame, period: int=DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc003_012；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

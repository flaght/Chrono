"""
因子代号: tc003_011
原历史名: tc003_011
因子定义: 收益率自相关衰减率：相邻两期收益率滚动自相关系数的变动率
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD=15
NAME="tc003_011"

def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    return df_lazy.with_columns(pl.rolling_cov(pl.col("high"),pl.col("value"),window_size=period).over("code").alias(NAME))

def compute(df_lazy: pl.LazyFrame, period: int=DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc003_011；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

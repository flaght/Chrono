"""
因子代号: tc003_004
原历史名: tc003_004
因子定义: 平均K线实体动量：Heikin-Ashi 实体高度在 period 周期内的滚动均值
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD=15
NAME="tc003_004"

def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    return df_lazy.with_columns(pl.col("value").rolling_skew(period).over("code").alias(NAME))

def compute(df_lazy: pl.LazyFrame, period: int=DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc003_004；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

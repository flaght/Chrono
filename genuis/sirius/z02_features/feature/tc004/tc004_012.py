"""
因子代号: tc004_012
原历史名: tc004_012
因子定义: 多空成交额相对失衡：基于收盘位置切分的多空成交额差值比率
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD = 15
NAME = "tc004_012"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(safe_div(2*(pl.col("high")-pl.col("low"))-(pl.col("close")-pl.col("open")).abs(),pl.col("value")/1_000_000).alias("_illiq"))
    return x.with_columns(pl.col("_illiq").rolling_skew(period).over("code").alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc004_012；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])

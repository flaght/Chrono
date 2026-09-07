"""
因子代号: tc002_002
原历史名: tc002_002
因子定义: 非流动性变异系数：收益率与成交量之比在 period 周期的滚动变异系数 (std / mean)
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_002"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(log_return().alias("_r"))
    x=x.with_columns(safe_div(pl.col("_r"),pl.col("volume")/10_000_000).alias("_illiq"))
    return x.with_columns(safe_div(pl.col("_illiq").rolling_std(period).over("code"),pl.col("_illiq").rolling_mean(period).over("code")).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_002；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

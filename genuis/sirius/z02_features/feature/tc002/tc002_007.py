"""
因子代号: tc002_007
原历史名: tc002_007
因子定义: 高价位成交量集中度：价格位于前20%最高区间内的成交量占总成交量的比重
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_007"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(log_return().alias("_r"))
    x=x.with_columns(pl.col("_r").rolling_mean(period).over("code").alias("_m"))
    x=x.with_columns((pl.col("_r")-pl.col("_m")).pow(2).alias("_sq"))
    return x.with_columns(safe_div(pl.when(pl.col("_r")>=pl.col("_m")).then(pl.col("_sq")).otherwise(0.0).rolling_sum(period).over("code"),pl.col("_sq").rolling_sum(period).over("code")).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_007；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

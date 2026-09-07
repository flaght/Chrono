"""
因子代号: tc002_016
原历史名: tc002_016
因子定义: 成交量峰值冲击效应：成交量超过均值1.5倍标准差时的收益率累积冲击
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_016"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(log_return().alias("_r"))
    x=x.with_columns(pl.col("_r").rolling_quantile(0.2,window_size=period).over("code").alias("_q"))
    return x.with_columns(pl.when(pl.col("_r")<pl.col("_q")).then(-pl.col("_r")).otherwise(0.0).rolling_mean(period).over("code").alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_016；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

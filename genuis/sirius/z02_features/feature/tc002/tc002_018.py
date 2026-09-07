"""
因子代号: tc002_018
原历史名: tc002_018
因子定义: 波动率调整成交量比率：成交量相对于滚动波动率标准化调整后的动量
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_018"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(((pl.col("open")+pl.col("high")+pl.col("low")+pl.col("close"))/4).alias("_twap"))
    x=x.with_columns(pl.col("_twap").rolling_mean(period).over("code").alias("_tm1"),pl.col("high").rolling_max(period).over("code").alias("_h1"),pl.col("low").rolling_min(period).over("code").alias("_l1"))
    x=x.with_columns(pl.col("_tm1").rolling_mean(period).over("code").alias("_tm2"),pl.col("_h1").rolling_max(period).over("code").alias("_h2"),pl.col("_l1").rolling_min(period).over("code").alias("_l2"))
    return x.with_columns((safe_div(pl.col("_tm2")-pl.col("_l2"),pl.col("_h2")-pl.col("_tm2"))+0.0001).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_018；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

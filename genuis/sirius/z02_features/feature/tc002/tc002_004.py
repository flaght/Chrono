"""
因子代号: tc002_004
原历史名: tc002_004
因子定义: 量价符号一致性相关系数：收益率绝对值与收益率符号加权成交量的滚动协方差比值
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_004"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(pl.col("close").rolling_quantile(0.8,window_size=period).over("code").alias("_q"))
    return x.with_columns(pl.when(pl.col("close")>=pl.col("_q")).then(pl.col("high")-pl.col("low")).otherwise(0.0).rolling_sum(period).over("code").alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_004；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

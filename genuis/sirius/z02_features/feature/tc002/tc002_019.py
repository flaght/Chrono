"""
因子代号: tc002_019
原历史名: tc002_019
因子定义: 量价变异系数比：价格收益变异系数与成交量变异系数的比值
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_019"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(log_return().alias("_r"))
    x=x.with_columns(pl.when(pl.col("_r")>0).then(pl.col("close")).alias("_p"),pl.when(pl.col("_r")<0).then(pl.col("close")).alias("_n"))
    x=x.with_columns(pl.col("_p").rolling_mean(period).over("code").alias("_pm"),pl.col("_n").rolling_mean(period).over("code").alias("_nm"))
    x=x.with_columns(safe_div(pl.col("_pm"),pl.col("_nm")).alias("_ratio"))
    x=x.with_columns(pl.col("_ratio").rolling_mean(period).over("code").alias("_m"),pl.col("_ratio").rolling_median(period).over("code").alias("_med"),pl.col("_ratio").rolling_max(period).over("code").alias("_max"),pl.col("_ratio").rolling_min(period).over("code").alias("_min"))
    return x.with_columns((-safe_div(pl.col("_m")-pl.col("_med"),pl.col("_max")-pl.col("_min"))).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_019；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

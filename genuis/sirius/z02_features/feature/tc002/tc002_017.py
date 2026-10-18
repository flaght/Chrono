"""
因子代号: tc002_017
原历史名: tc002_017
因子定义: 量价动量协同指数：价格动量与成交量动量的加权几何协同强度
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_017"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(safe_div(pl.col("value"),pl.col("volume")).alias("_vwap"))
    x=x.with_columns(pl.col("_vwap").rolling_mean(period).over("code").alias("_mean"),safe_div((pl.col("_vwap")*pl.col("volume")).rolling_sum(period).over("code"),pl.col("volume").rolling_sum(period).over("code")).alias("_weighted"))
    return x.with_columns(pl.when((pl.col("_mean")>0)&(pl.col("_weighted")>0)).then((pl.col("_mean")/pl.col("_weighted")).log()).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_017；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

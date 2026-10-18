"""
因子代号: tc002_012
原历史名: tc002_012
因子定义: 量价背离幅度：价格变动率绝对值与成交量变动率绝对值的相对比率
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_012"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(log_return().alias("_r"))
    x=x.with_columns(pl.col("_r").rolling_mean(period).over("code").alias("_m"),pl.col("_r").rolling_std(period).over("code").alias("_s"))
    x=x.with_columns(pl.when(pl.col("_r")>pl.col("_m")+pl.col("_s")).then(pl.col("volume")).otherwise(0.0).alias("_v"),pl.when(pl.col("_r")>pl.col("_m")+pl.col("_s")).then(pl.col("_r")).otherwise(0.0).alias("_er"))
    x=x.with_columns(safe_div(pl.col("_v").rolling_mean(period).over("code"),pl.col("volume").rolling_mean(period).over("code")).alias("_share"))
    return x.with_columns((pl.col("_er").rolling_std(period).over("code")*pl.col("_share")).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_012；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

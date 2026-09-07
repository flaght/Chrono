"""
因子代号: tc004_004
原历史名: tc004_004
因子定义: 价格区间二阶矩波动比：高开低收二阶矩综合波动率相对均值的偏离
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD = 15
NAME = "tc004_004"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(log_return().alias("_r"),(pl.col("high").rolling_sum(period).over("code")+pl.col("open").rolling_sum(period).over("code")+pl.col("low").rolling_sum(period).over("code")+pl.col("close").rolling_sum(period).over("code")).alias("_s1"),(pl.col("high").pow(2).rolling_sum(period).over("code")+pl.col("open").pow(2).rolling_sum(period).over("code")+pl.col("low").pow(2).rolling_sum(period).over("code")+pl.col("close").pow(2).rolling_sum(period).over("code")).alias("_s2"))
    x=x.with_columns((safe_div(pl.col("_s2"),pl.col("high")+pl.col("open")+pl.col("low")+pl.col("close"))-safe_div(pl.col("_s1"),pl.col("high")+pl.col("open")+pl.col("low")+pl.col("close")).pow(2)).sqrt().alias("_std"),pl.col("_s1").rolling_mean(period).over("code").alias("_mean"))
    x=x.with_columns(safe_div(pl.col("_std"),pl.col("_mean")).alias("_better"))
    x=x.with_columns(pl.col("_better").rolling_mean(period).over("code").alias("_bm"),pl.col("_better").rolling_std(period).over("code").alias("_bs"))
    x=x.with_columns(pl.when(pl.col("_better")>pl.col("_bm")+pl.col("_bs")).then(pl.col("_better")).alias("_higher"))
    return x.with_columns(pl.rolling_cov(pl.col("_higher"),safe_div(pl.col("_r"),pl.col("_better")),window_size=period).over("code").forward_fill().over("code").fill_null(0.0).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc004_004；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])

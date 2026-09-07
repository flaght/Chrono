"""
因子代号: tc004_013
原历史名: tc004_013
因子定义: 日内振幅均线突破度：日内振幅 (high-low)/open 相对其移动平均的偏离度
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD = 15
NAME = "tc004_013"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(pl.col("volume").rolling_mean(period).over("code").alias("_vm"),pl.col("close").rolling_mean(period).over("code").alias("_cm"))
    return x.with_columns((safe_div(pl.col("_cm"),pl.col("_cm").shift(period).over("code"))*safe_div(pl.col("_vm"),pl.col("_vm").shift(period).over("code"))).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc004_013；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

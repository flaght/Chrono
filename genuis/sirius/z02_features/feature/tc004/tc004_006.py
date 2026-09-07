"""
因子代号: tc004_006
原历史名: tc004_006
因子定义: 成交额与价格极差相关系数：成交额与日内 (high - low) 的滚动相关性
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD = 15
NAME = "tc004_006"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(pl.when((pl.col("close")-pl.col("open")).abs()<=0.5*(pl.col("high")-pl.col("low"))).then(pl.col("volume")).otherwise(0.0).alias("_v"))
    return x.with_columns(safe_div(pl.col("_v").rolling_sum(period).over("code"),pl.col("volume").rolling_sum(period).over("code")).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc004_006；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])

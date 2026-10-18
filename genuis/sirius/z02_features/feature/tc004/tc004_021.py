"""
因子代号: tc004_021
原历史名: tc004_021
因子定义: 价格极值区间压缩比：价格最高最低区间与典型波动区间的比率
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD = 15
NAME = "tc004_021"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(safe_div(pl.col("volume"),pl.col("volume").rolling_sum(period).over("code")).alias("_w"))
    x=x.with_columns(safe_div(pl.rolling_cov(pl.col("low")*pl.col("_w"),pl.col("high")*pl.col("_w"),window_size=period).over("code"),(pl.col("low")*pl.col("_w")).rolling_var(period).over("code")).alias("_beta"),pl.rolling_corr(pl.col("low")*pl.col("_w"),pl.col("high")*pl.col("_w"),window_size=period).over("code").pow(2).alias("_r2"))
    x=x.with_columns(safe_div(pl.col("_beta")-pl.col("_beta").rolling_mean(period).over("code"),pl.col("_beta").rolling_std(period).over("code")).alias("_z"))
    return x.with_columns((pl.col("_beta")*pl.col("_z")*pl.col("_r2")).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc004_021；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

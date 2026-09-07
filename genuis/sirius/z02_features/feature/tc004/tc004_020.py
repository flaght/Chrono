"""
因子代号: tc004_020
原历史名: tc004_020
因子定义: 成交量加权日内收益率：以成交量为权重的 (close - open) 滚动加权和
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD = 15
NAME = "tc004_020"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(safe_div(pl.rolling_cov(pl.col("low"),pl.col("high"),window_size=period).over("code"),pl.col("low").rolling_var(period).over("code")).alias("_beta"),pl.rolling_corr(pl.col("low"),pl.col("high"),window_size=period).over("code").pow(2).alias("_r2"))
    x=x.with_columns(safe_div(pl.col("_beta")-pl.col("_beta").rolling_mean(period).over("code"),pl.col("_beta").rolling_std(period).over("code")).alias("_z"))
    return x.with_columns((pl.col("_beta")*pl.col("_z")*pl.col("_r2")).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc004_020；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

"""
因子代号: tc003_007
原历史名: tc003_007
因子定义: 波动率加权成交量动量：成交量变化率经收益率波动率调整后的动量
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD=15
NAME="tc003_007"

def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(pl.col("open").diff(2).over("code").alias("_od"),pl.col("high").diff(2).over("code").alias("_hd"),pl.col("close").diff(2).over("code").alias("_cd"),pl.col("volume").diff(2).over("code").alias("_vd"),pl.col("value").diff(2).over("code").alias("_xd"))
    x=x.with_columns(pl.rolling_corr(pl.col("_od"),pl.col("_vd"),window_size=period).over("code").alias("_c1"),pl.rolling_corr(pl.col("_hd"),pl.col("_xd"),window_size=period).over("code").alias("_c2"))
    return x.with_columns((pl.rolling_corr(1/(1+(-pl.col("_cd")).exp()),pl.col("_c1"),window_size=period).over("code")-pl.col("_c2")).alias(NAME))

def compute(df_lazy: pl.LazyFrame, period: int=DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc003_007；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

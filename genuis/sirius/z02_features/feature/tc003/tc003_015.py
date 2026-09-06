"""
因子代号: tc003_015
原历史名: tc003_015
因子定义: 成交量加权典型价格偏离：典型价格相对成交量加权均价的滚动偏离度
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD=15
NAME="tc003_015"

def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(safe_div(pl.rolling_cov(pl.col("volume"),pl.col("close").sqrt(),window_size=period).over("code"),pl.col("volume").rolling_std(period).over("code").pow(2)).alias("_c1"))
    x=x.with_columns((1/(1+(-pl.col("_c1")).exp())).alias("_s"))
    x=x.with_columns(safe_div(pl.rolling_cov(pl.col("_s"),pl.col("low"),window_size=period).over("code"),pl.col("_s").rolling_std(period).over("code")).alias("_c2"))
    return x.with_columns((pl.col("low")-pl.col("_s")*pl.col("_c2")).alias(NAME))

def compute(df_lazy: pl.LazyFrame, period: int=DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc003_015；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

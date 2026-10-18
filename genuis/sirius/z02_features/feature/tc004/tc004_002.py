"""
因子代号: tc004_002
原历史名: tc004_002
因子定义: 成交额加权标准化收益：以成交额为权重的收益率/波动率标准化均值
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD = 15
NAME = "tc004_002"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(log_return().alias("_r"),pl.col("close").rolling_std(period).over("code").alias("_s"))
    return x.with_columns(safe_div((pl.col("value")*safe_div(pl.col("_r"),pl.col("_s"))).rolling_sum(period).over("code"),pl.col("value").rolling_sum(period).over("code")).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc004_002；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])

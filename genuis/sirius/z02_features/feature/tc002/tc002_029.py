"""
因子代号: tc002_029
原历史名: tc002_029
因子定义: 成交量极值不对称性：放量周期收益率与缩量周期收益率的均值差异
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_029"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(log_return().alias("_r"))
    x=x.with_columns(pl.col("_r").rolling_sum(period).over("code").alias("_sum"))
    x=x.with_columns(pl.col("_sum").rolling_max(period).over("code").alias("_peak"))
    return x.with_columns((pl.col("_sum")-pl.col("_peak")).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_029；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

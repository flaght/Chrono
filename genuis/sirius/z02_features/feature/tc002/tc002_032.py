"""
因子代号: tc002_032
原历史名: tc002_032
因子定义: 成交量波动率聚集度：成交量波动率相对历史均值的偏离与聚集程度
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_032"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(log_return().alias("_r"))
    x=x.with_columns(pl.col("_r").pow(2).alias("_rv"),pl.when(pl.col("_r")>0).then(pl.col("_r").pow(2)).otherwise(0.0).alias("_up"),pl.when(pl.col("_r")<0).then(pl.col("_r").pow(2)).otherwise(0.0).alias("_down"))
    return x.with_columns(safe_div((pl.col("_up")-pl.col("_down")).rolling_sum(period).over("code"),pl.col("_rv").rolling_sum(period).over("code")).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_032；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

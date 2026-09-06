"""
因子代号: tc002_003
原历史名: tc002_003
因子定义: 成交占比信息熵：以滚动窗口内成交量占比为概率分布计算信息熵 rolling_sum(p * log(p))
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_003"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(safe_div(pl.col("volume"),pl.col("volume").rolling_sum(period).over("code")).alias("_share"))
    x=x.with_columns(pl.when(pl.col("_share")>0).then(pl.col("_share")*pl.col("_share").log()).alias("_entropy"))
    return x.with_columns(pl.col("_entropy").rolling_sum(period).over("code").alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_003；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

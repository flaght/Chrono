"""
因子代号: tc002_001
原历史名: tc002_001
因子定义: 成交量突破事件加权滚动和：当成交量超过 (均值+1倍标准差) 时计入事件指示，在 period 周期内累计 (成交量 + 事件指示)
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_001"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(pl.col("volume").rolling_mean(period).over("code").alias("_m"),pl.col("volume").rolling_std(period).over("code").alias("_s"))
    x=x.with_columns((pl.col("volume")+pl.when(pl.col("volume")>pl.col("_m")+pl.col("_s")).then(1.0).otherwise(0.0)).alias("_event_volume"))
    return x.with_columns(pl.col("_event_volume").rolling_sum(period).over("code").alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_001；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

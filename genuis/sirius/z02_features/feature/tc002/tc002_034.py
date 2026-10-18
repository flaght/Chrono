"""
因子代号: tc002_034
原历史名: tc002_034
因子定义: 收盘价均线趋势度：收盘价相对短期与长期移动平均线综合偏离度
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_034"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(pl.col("volume").rolling_quantile(0.8,window_size=period).over("code").alias("_q"))
    return x.with_columns(pl.when(pl.col("volume")>pl.col("_q")).then(pl.col("volume")).otherwise(0.0).rolling_sum(period).over("code").alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_034；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

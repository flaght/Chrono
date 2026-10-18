"""
因子代号: tc002_023
原历史名: tc002_023
因子定义: 成交量加速度动量：成交量二阶差分相对移动平均的加速强度
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_023"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(pl.col("close").rolling_std(period).over("code").alias("_ps"),pl.col("volume").rolling_std(period).over("code").alias("_vs"))
    return x.with_columns((pl.rolling_corr(pl.col("_ps"),pl.col("_vs"),window_size=period).over("code")*safe_div(pl.col("_ps"),pl.col("_vs"))).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_023；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

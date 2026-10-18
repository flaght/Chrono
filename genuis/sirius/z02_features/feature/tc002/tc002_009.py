"""
因子代号: tc002_009
原历史名: tc002_009
因子定义: 累计成交量波动率：累计成交量在 period 周期内的滚动标准差
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_009"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(((pl.col("open")+pl.col("high")+pl.col("low")+pl.col("close"))/4).alias("_twap"),safe_div(pl.col("value"),pl.col("volume")).alias("_vwap"))
    return x.with_columns(safe_div(pl.col("_twap").rolling_mean(period).over("code"),pl.col("_vwap").rolling_mean(period).over("code")).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_009；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

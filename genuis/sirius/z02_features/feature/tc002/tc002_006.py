"""
因子代号: tc002_006
原历史名: tc002_006
因子定义: 极端高成交量区间的价格偏离度：最高20%成交量区间的加权价格相对均价的偏离
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_006"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(((pl.col("open")+pl.col("high")+pl.col("low")+pl.col("close"))/4).alias("_twap"))
    return x.with_columns(safe_div(pl.col("_twap")-pl.col("low"),pl.col("high")-pl.col("low")).rolling_mean(period).over("code").alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_006；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

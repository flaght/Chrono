"""
因子代号: tc002_028
原历史名: tc002_028
因子定义: 价格与成交量滚动偏度差：收益率偏度与成交量偏度的差值
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_028"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(pl.when((pl.col("volume")>0)&(pl.col("volume").shift(1).over("code")>0)).then((pl.col("volume")/pl.col("volume").shift(1).over("code")).log()).alias("_vchg"))
    return x.with_columns(pl.when(pl.col("_vchg")>0).then(pl.col("_vchg")).rolling_mean(period).over("code").alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_028；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

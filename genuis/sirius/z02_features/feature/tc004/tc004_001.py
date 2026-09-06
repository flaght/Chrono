"""
因子代号: tc004_001
原历史名: tc004_001
因子定义: 成交额占比信息熵：成交额占滚动总成交额比重的滞后自信息量
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD = 15
NAME = "tc004_001"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(safe_div(pl.col("value"),pl.col("value").rolling_sum(period).over("code")).alias("_p"))
    x=x.with_columns(pl.when((pl.col("_p")>0)&(pl.col("_p").shift(1).over("code")>0)).then((pl.col("_p")/pl.col("_p").shift(1).over("code")).log()).alias("_chg"))
    return x.with_columns((-pl.col("_p")*pl.col("_chg")).rolling_sum(period).over("code").alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc004_001；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])

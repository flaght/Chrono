"""
因子代号: tc004_019
原历史名: tc004_019
因子定义: 高低价振幅标准差：日内价格相对振幅在 period 周期内的波动率
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD = 15
NAME = "tc004_019"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    return df_lazy.with_columns(pl.col("close").rolling_map(lambda s: float(s[-1]+(s[-1]-s[0])/max(len(s)-1,1)),window_size=period).over("code").alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc004_019；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

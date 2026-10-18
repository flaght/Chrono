"""
因子代号: tc002_027
原历史名: tc002_027
因子定义: 成交量标准化收益惯性：经成交量标准差归一化后的收益率滚动自相关性
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_027"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(log_return().alias("_r"),(pl.col("volume")/1_000_000).alias("_v"))
    x=x.with_columns(pl.col("_r").rolling_kurtosis(period).over("code").alias("_rk"),pl.col("_v").rolling_kurtosis(period).over("code").alias("_vk"))
    return x.with_columns(pl.rolling_corr(pl.col("_rk"),pl.col("_vk"),window_size=period).over("code").alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_027；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

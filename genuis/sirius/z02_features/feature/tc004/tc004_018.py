"""
因子代号: tc004_018
原历史名: tc004_018
因子定义: 价格二阶差分动量：价格变动加速度在 period 周期内的平滑指标
"""
import polars as pl
from feature.utils.common import log_return, safe_div, validate_period
DEFAULT_PERIOD = 15
NAME = "tc004_018"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """计算单组周期参数。"""
    x=df_lazy.with_columns(log_return().alias("_c"),log_return("volume").alias("_v"))
    return x.with_columns(safe_div((pl.col("_c")*pl.col("_v")).rolling_mean(period).over("code"),pl.col("_c").rolling_std(period).over("code")).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc004_018；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time","code"]),period).select(["trade_time","code",NAME])

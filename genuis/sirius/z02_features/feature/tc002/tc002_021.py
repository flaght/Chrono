"""
因子代号: tc002_021
原历史名: tc002_021
因子定义: 量价秩相关系数：价格收益率排名与成交量排名的滚动 Spearman 秩相关
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_021"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(log_return().alias("_r"))
    x=x.with_columns(pl.col("_r").abs().alias("_a"),(pl.col("_r").sign()*pl.col("volume")).alias("_sv"))
    return x.with_columns((pl.rolling_corr(pl.col("_a"),pl.col("_sv"),window_size=period).over("code")*safe_div(pl.col("_a").rolling_std(period).over("code"),pl.col("_sv").rolling_std(period).over("code"))).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_021；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

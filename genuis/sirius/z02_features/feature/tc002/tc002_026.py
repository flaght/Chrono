"""
因子代号: tc002_026
原历史名: tc002_026
因子定义: 量价趋势共振得分：价格多头排列与成交量放大趋势的双重确认共振得分
"""
import polars as pl

from feature.utils import log_return, safe_div, validate_period

DEFAULT_PERIOD = 15
NAME = "tc002_026"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code 及公式所需基础字段计算单组周期参数。"""
    x=df_lazy.with_columns(log_return().alias("_r"),(pl.col("volume")/1_000_000).alias("_v"))
    x=x.with_columns(pl.col("_r").rolling_mean(period).over("code").alias("_rm"),pl.col("_r").rolling_std(period).over("code").alias("_rs"),pl.col("_v").rolling_mean(period).over("code").alias("_vm"),pl.col("_v").rolling_std(period).over("code").alias("_vs"))
    x=x.with_columns(pl.rolling_corr(pl.col("_rs"),pl.col("_vs"),window_size=period).over("code").alias("_c"))
    return x.with_columns((safe_div(pl.col("_rm"),pl.col("_vm"))+pl.col("_c")*safe_div(pl.col("_rs"),pl.col("_vs"))*safe_div(pl.col("_vs"),pl.col("_vm")).pow(2)).alias(NAME))


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc002_026；输入含 trade_time、code、open、high、low、close、volume、value。"""
    validate_period(period)
    result_lazy = calculate(df_lazy.sort(["trade_time", "code"]), period)
    return result_lazy.select(["trade_time", "code", NAME])

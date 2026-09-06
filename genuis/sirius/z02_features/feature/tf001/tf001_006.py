"""
因子代号: tf001_006
原历史名: tf001_006
因子定义: 持仓流动性指标 1：持仓量变动相对价格波动的流动性比率
"""
import polars as pl

from feature.utils.common import validate_period

DEFAULT_PERIOD = 15
NAME = "tf001_006"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、openint 计算 tf001_006 核心公式。"""
    return (
        df_lazy
        .with_columns(
            (pl.col("openint") / 1e6).alias("_scaled_openint")
        )
        .with_columns(
            pl.col("_scaled_openint")
            .shift(1)
            .over("code")
            .alias("_previous_scaled_openint")
        )
        .with_columns(
            pl.when(
                (pl.col("_scaled_openint") > 0)
                & (pl.col("_previous_scaled_openint") > 0)
            )
            .then(
                (
                    pl.col("_scaled_openint")
                    / pl.col("_previous_scaled_openint")
                ).log()
            )
            .otherwise(None)
            .alias("_openint_log_return")
        )
        .with_columns(
            pl.col("_openint_log_return")
            .rolling_mean(period)
            .over("code")
            .alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf001_006；输入必须包含 trade_time、code、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])

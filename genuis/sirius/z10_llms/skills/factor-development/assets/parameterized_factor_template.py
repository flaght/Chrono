"""因子定义: FACTOR_DEFINITION"""
import polars as pl


DEFAULT_PERIOD = 20
NAME = "FACTOR_NAME"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用 trade_time、code、REQUIRED_COLUMNS 计算 FACTOR_NAME 核心公式。"""
    return (
        df_lazy
        .with_columns(
            FACTOR_EXPRESSION.alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(
    df_lazy: pl.LazyFrame,
    period: int = DEFAULT_PERIOD,
) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 FACTOR_NAME；输入必须包含 trade_time、code、REQUIRED_COLUMNS。"""
    if not isinstance(period, int) or isinstance(period, bool) or period <= 0:
        raise ValueError("period 必须是正整数")
    return calculate(df_lazy.sort(["trade_time", "code"]), period)

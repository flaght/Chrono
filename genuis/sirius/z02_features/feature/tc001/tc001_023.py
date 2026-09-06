"""
因子代号: tc001_023
原历史名: tc001_023
因子定义: MACD异同移动平均指标 = fast周期EMA - slow周期EMA
"""
import polars as pl


DEFAULT_PERIODS = (12, 26)


def calculate(
    periods: tuple[int, int],
) -> pl.Expr:
    """
    实现一组快慢周期对应的MACD表达式。

    参数:
        periods: (fast, slow) 二元组，要求 fast < slow
    返回:
        pl.Expr: 快周期EMA减慢周期EMA的表达式
    """
    fast, slow = periods
    return (
        pl.col("close").ewm_mean(span=fast).over("code")
        - pl.col("close").ewm_mean(span=slow).over("code")
    )


def compute(
    df_lazy: pl.LazyFrame,
    periods: tuple[int, int] = DEFAULT_PERIODS,
) -> pl.LazyFrame:
    """
    对外构造 tc001_023 因子计算图，未传周期时使用 DEFAULT_PERIODS。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
        periods: (fast, slow) 二元组，默认使用 DEFAULT_PERIODS
    返回:
        pl.LazyFrame: 包含 trade_time、code 和 ta023_<fast>_<slow> 因子列
    """
    if len(periods) != 2:
        raise ValueError("periods 必须是 (fast, slow) 二元组")

    fast, slow = periods
    if any(not isinstance(period, int) or isinstance(period, bool) or period <= 0
           for period in periods):
        raise ValueError("fast 和 slow 必须为正整数")
    if fast >= slow:
        raise ValueError("periods 必须满足 fast < slow")

    factor_name = f"tc001_023_{fast}_{slow}"
    return (
        df_lazy
        .sort(by=["trade_time", "code"])
        .with_columns(calculate(periods).alias(factor_name))
        .select(["trade_time", "code", factor_name])
    )

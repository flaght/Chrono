"""
因子代号: tc001_025
原历史名: tc001_025
因子定义: N周期价格动量 = close / close.shift(N) - 1
"""
from collections.abc import Iterable

import polars as pl


DEFAULT_WINDOWS = (5, 10)


def calculate(
    window: int,
) -> pl.Expr:
    """
    实现单个窗口的价格动量表达式。

    参数:
        window: 正整数周期
    返回:
        pl.Expr: 当前收盘价相对指定周期前收盘价的收益率表达式
    """
    previous_close = pl.col("close").shift(window).over("code")
    return (
        pl.when(previous_close.is_null() | (previous_close == 0))
        .then(None)
        .otherwise(pl.col("close") / previous_close - 1)
    )


def compute(
    df_lazy: pl.LazyFrame,
    windows: Iterable[int] = DEFAULT_WINDOWS,
) -> pl.LazyFrame:
    """
    对外构造 tc001_025 因子计算图，未传窗口时使用 DEFAULT_WINDOWS。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
        windows: 动量周期，默认使用 DEFAULT_WINDOWS
    返回:
        pl.LazyFrame: 包含 trade_time、code 和 ta025_<周期> 因子列
    """
    normalized_windows = tuple(dict.fromkeys(windows))
    if not normalized_windows:
        raise ValueError("至少需要一个动量窗口")
    if any(not isinstance(window, int) or isinstance(window, bool) or window <= 0
           for window in normalized_windows):
        raise ValueError("动量窗口必须为正整数")

    factor_names = [f"tc001_025_{window}" for window in normalized_windows]
    expressions = [
        calculate(window).alias(factor_name)
        for window, factor_name in zip(normalized_windows, factor_names)
    ]

    return (
        df_lazy
        .sort(by=["trade_time", "code"])
        .with_columns(expressions)
        .select(["trade_time", "code", *factor_names])
    )

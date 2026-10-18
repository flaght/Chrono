"""基于原生 Polars Expr 的因子挖掘原子算子库。

算子直接接收并返回 ``pl.Expr``，不构建公式树。若窗口结果还要进入下一层
窗口，调用方必须先用 ``with_columns`` 将结果落为临时列。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

import polars as pl

from .common import rolling_rank, rolling_sign_change_rate


ExprLike = Union[str, int, float, pl.Expr]
Operator = Callable[..., pl.Expr]


def as_expr(value: ExprLike) -> pl.Expr:
    if isinstance(value, str):
        return pl.col(value)
    if isinstance(value, pl.Expr):
        return value
    return pl.lit(value)


def col(name: str) -> pl.Expr:
    return pl.col(name)


def safe_div(numerator: ExprLike, denominator: ExprLike) -> pl.Expr:
    left, right = as_expr(numerator), as_expr(denominator)
    return pl.when(right.is_not_null() & (right != 0)).then(
        left / right
    ).otherwise(None)


def abs_(value: ExprLike) -> pl.Expr:
    return as_expr(value).abs()


def log(value: ExprLike) -> pl.Expr:
    expression = as_expr(value)
    return pl.when(expression > 0).then(expression.log()).otherwise(None)


def exp(value: ExprLike) -> pl.Expr:
    return as_expr(value).exp()


def atan(value: ExprLike) -> pl.Expr:
    return as_expr(value).arctan()


def tanh(value: ExprLike) -> pl.Expr:
    return as_expr(value).tanh()


def sign(value: ExprLike) -> pl.Expr:
    expression = as_expr(value)
    return pl.when(expression > 0).then(1.0).when(
        expression < 0
    ).then(-1.0).otherwise(0.0)


def clip(value: ExprLike, lower: float, upper: float) -> pl.Expr:
    return as_expr(value).clip(lower, upper)


def minimum(left: ExprLike, right: ExprLike) -> pl.Expr:
    return pl.min_horizontal(as_expr(left), as_expr(right))


def maximum(left: ExprLike, right: ExprLike) -> pl.Expr:
    return pl.max_horizontal(as_expr(left), as_expr(right))


def where(condition: ExprLike, value: ExprLike, otherwise: ExprLike = 0.0) -> pl.Expr:
    return pl.when(as_expr(condition)).then(as_expr(value)).otherwise(
        as_expr(otherwise)
    )


def lag(value: ExprLike, periods: int = 1, group_by: str = "code") -> pl.Expr:
    return as_expr(value).shift(periods).over(group_by)


def diff(value: ExprLike, periods: int = 1, group_by: str = "code") -> pl.Expr:
    return as_expr(value).diff(periods).over(group_by)


def pct_change(value: ExprLike, periods: int = 1, group_by: str = "code") -> pl.Expr:
    current = as_expr(value)
    return safe_div(current, current.shift(periods).over(group_by)) - 1.0


def log_return(value: ExprLike, periods: int = 1, group_by: str = "code") -> pl.Expr:
    current = as_expr(value)
    return log(safe_div(current, current.shift(periods).over(group_by)))


def ts_sum(value: ExprLike, period: int, group_by: str = "code") -> pl.Expr:
    return as_expr(value).rolling_sum(period).over(group_by)


def ts_mean(value: ExprLike, period: int, group_by: str = "code") -> pl.Expr:
    return as_expr(value).rolling_mean(period).over(group_by)


def ts_std(value: ExprLike, period: int, group_by: str = "code") -> pl.Expr:
    return as_expr(value).rolling_std(period).over(group_by)


def ts_min(value: ExprLike, period: int, group_by: str = "code") -> pl.Expr:
    return as_expr(value).rolling_min(period).over(group_by)


def ts_max(value: ExprLike, period: int, group_by: str = "code") -> pl.Expr:
    return as_expr(value).rolling_max(period).over(group_by)


def ts_median(value: ExprLike, period: int, group_by: str = "code") -> pl.Expr:
    return as_expr(value).rolling_median(period).over(group_by)


def ts_skew(value: ExprLike, period: int, group_by: str = "code") -> pl.Expr:
    return as_expr(value).rolling_skew(period).over(group_by)


def ts_kurtosis(value: ExprLike, period: int, group_by: str = "code") -> pl.Expr:
    return as_expr(value).rolling_kurtosis(period).over(group_by)


def ts_quantile(value: ExprLike, quantile: float, period: int, group_by: str = "code") -> pl.Expr:
    return as_expr(value).rolling_quantile(
        quantile, window_size=period
    ).over(group_by)


def ts_rank(value: ExprLike, period: int, *, pct: bool = True, group_by: str = "code") -> pl.Expr:
    return rolling_rank(as_expr(value), period, pct=pct).over(group_by)


def ts_corr(left: ExprLike, right: ExprLike, period: int, group_by: str = "code") -> pl.Expr:
    return pl.rolling_corr(
        as_expr(left), as_expr(right), window_size=period
    ).over(group_by)


def ts_cov(left: ExprLike, right: ExprLike, period: int, group_by: str = "code") -> pl.Expr:
    return pl.rolling_cov(
        as_expr(left), as_expr(right), window_size=period
    ).over(group_by)


def ts_sign_change_rate(value: ExprLike, period: int, group_by: str = "code") -> pl.Expr:
    return rolling_sign_change_rate(as_expr(value), period).over(group_by)


def ewm_mean(value: ExprLike, span: int, group_by: str = "code") -> pl.Expr:
    return as_expr(value).ewm_mean(span=span).over(group_by)


def materialize(df_lazy: pl.LazyFrame, expression: pl.Expr, name: str) -> pl.LazyFrame:
    """将算子结果落列，供下一层算子通过 ``pl.col(name)`` 使用。"""
    return df_lazy.with_columns(expression.alias(name))


UNARY_OPERATORS: dict[str, Operator] = {
    "abs": abs_, "log": log, "exp": exp, "atan": atan,
    "tanh": tanh, "sign": sign,
}
BINARY_OPERATORS: dict[str, Operator] = {
    "safe_div": safe_div, "minimum": minimum, "maximum": maximum,
}
CONDITIONAL_OPERATORS: dict[str, Operator] = {
    "clip": clip, "where": where,
}
TIME_SERIES_OPERATORS: dict[str, Operator] = {
    "lag": lag, "diff": diff, "pct_change": pct_change,
    "log_return": log_return, "ts_sum": ts_sum, "ts_mean": ts_mean,
    "ts_std": ts_std, "ts_min": ts_min, "ts_max": ts_max,
    "ts_median": ts_median, "ts_skew": ts_skew,
    "ts_kurtosis": ts_kurtosis, "ts_quantile": ts_quantile,
    "ts_rank": ts_rank, "ts_corr": ts_corr, "ts_cov": ts_cov,
    "ts_sign_change_rate": ts_sign_change_rate, "ewm_mean": ewm_mean,
}
OPERATORS: dict[str, Operator] = {
    **UNARY_OPERATORS,
    **BINARY_OPERATORS,
    **CONDITIONAL_OPERATORS,
    **TIME_SERIES_OPERATORS,
}


__all__ = [
    "BINARY_OPERATORS", "CONDITIONAL_OPERATORS", "ExprLike", "OPERATORS", "Operator",
    "TIME_SERIES_OPERATORS", "UNARY_OPERATORS", "abs_", "as_expr",
    "atan", "clip", "col", "diff", "ewm_mean", "exp", "lag", "log",
    "log_return", "materialize", "maximum", "minimum", "pct_change",
    "safe_div", "sign", "tanh", "ts_corr", "ts_cov", "ts_kurtosis",
    "ts_max", "ts_mean", "ts_median", "ts_min", "ts_quantile",
    "ts_rank", "ts_sign_change_rate", "ts_skew", "ts_std", "ts_sum",
    "where",
]

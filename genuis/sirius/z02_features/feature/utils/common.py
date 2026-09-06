"""Feature packages shared Polars helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import polars as pl


KEY_COLUMNS = ["trade_time", "code", "symbol"]
FactorCompute = Callable[..., pl.LazyFrame]
NATIVE_ROLLING_RANK_VERSION = (1, 36, 1)


@dataclass(frozen=True)
class FactorSpec:
    """用于注册、筛选和校验因子的稳定元数据。"""

    name: str
    family: str
    market_scope: str
    data_granularity: str
    batch: str
    required_fields: tuple[str, ...]
    compute: FactorCompute
    source_name: str | None = None
    aliases: tuple[str, ...] = ()


def _version_tuple(version: str) -> tuple[int, int, int]:
    """提取主版本、次版本和补丁版本，兼容 rc/dev 等后缀。"""
    numbers: list[int] = []
    for component in version.split(".")[:3]:
        digits = ""
        for character in component:
            if not character.isdigit():
                break
            digits += character
        numbers.append(int(digits) if digits else 0)
    return tuple((numbers + [0, 0, 0])[:3])


def safe_div(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """分母为零或空值时返回空值。"""
    return pl.when(denominator.is_not_null() & (denominator != 0)).then(
        numerator / denominator
    ).otherwise(None)


def log_return(column: str = "close") -> pl.Expr:
    """按品种计算相邻周期对数收益。"""
    current = pl.col(column)
    previous = current.shift(1).over("code")
    return pl.when((current > 0) & (previous > 0)).then(
        (current / previous).log()
    ).otherwise(None)


def rolling_rank(
    expression: pl.Expr,
    period: int,
    *,
    pct: bool = False,
) -> pl.Expr:
    """返回窗口末值的平均名次，并兼容 Polars 1.36.1 之前的版本。"""
    if (
        _version_tuple(pl.__version__) >= NATIVE_ROLLING_RANK_VERSION
        and hasattr(expression, "rolling_rank")
    ):
        native_rank = expression.rolling_rank(
            window_size=period,
            method="average",
        )
        return native_rank / period if pct else native_rank

    def rank_last(values: pl.Series) -> float | None:
        if values.null_count() > 0:
            return None
        rank = float(values.rank(method="average")[-1])
        return rank / len(values) if pct else rank

    return expression.rolling_map(rank_last, window_size=period)


def rolling_sign_change_rate(expression: pl.Expr, period: int) -> pl.Expr:
    """计算窗口内相邻观测符号发生变化的比例。"""
    def sign_change_rate(values: pl.Series) -> float | None:
        items = values.to_list()
        if len(items) < 2 or any(value is None for value in items):
            return None
        signs = [(value > 0) - (value < 0) for value in items]
        changes = sum(left != right for left, right in zip(signs, signs[1:]))
        return changes / (len(signs) - 1)

    return expression.rolling_map(sign_change_rate, window_size=period)


def validate_period(period: int) -> None:
    if not isinstance(period, int) or isinstance(period, bool) or period <= 0:
        raise ValueError("period 必须为正整数")


def compute_factor_batch(
    df_lazy: pl.LazyFrame,
    factors: Mapping[str, FactorCompute],
    names: Iterable[str] | None = None,
    params: Mapping[str, Mapping[str, object]] | None = None,
    key_columns: Iterable[str] = KEY_COLUMNS,
) -> pl.LazyFrame:
    """Build and align selected factor computation graphs from a registry."""
    selected = list(factors) if names is None else list(names)
    if not selected:
        raise ValueError("至少需要指定一个因子")

    duplicate_names = sorted(
        name for name in set(selected) if selected.count(name) > 1)
    if duplicate_names:
        raise ValueError(f"因子名不能重复: {duplicate_names}")

    unknown_names = sorted(set(selected) - factors.keys())
    if unknown_names:
        raise KeyError(f"不存在的因子: {unknown_names}")

    factor_params = {} if params is None else dict(params)
    unknown_param_names = sorted(set(factor_params) - set(selected))
    if unknown_param_names:
        raise KeyError(f"参数指定了未选中的因子: {unknown_param_names}")

    frames = [
        factors[name](df_lazy, **dict(factor_params.get(name, {})))
        for name in selected
    ]
    if len(frames) == 1:
        return frames[0]
    sort_keys = [c for c in key_columns if c in frames[0].collect_schema().names()]
    return pl.concat(frames, how="align").sort(sort_keys)


__all__ = [
    "FactorCompute",
    "FactorSpec",
    "KEY_COLUMNS",
    "compute_factor_batch",
    "log_return",
    "rolling_rank",
    "rolling_sign_change_rate",
    "safe_div",
    "validate_period",
]

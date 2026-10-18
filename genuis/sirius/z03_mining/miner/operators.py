"""三个因子挖掘后端共用的算子分类、配置校验与注册表适配。"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from feature.utils import formula as formula_ops

DEFAULT_UNARY_OPERATORS = ("abs", "log", "atan", "tanh", "sign")
DEFAULT_WINDOW_OPERATORS = (
    "lag", "diff", "pct_change", "log_return", "ts_sum", "ts_mean",
    "ts_std", "ts_min", "ts_max", "ts_median", "ts_rank",
    "ts_skew", "ts_kurtosis", "ts_sign_change_rate", "ewm_mean",
)
DEFAULT_BINARY_OPERATORS = ("safe_div", "minimum", "maximum")
DEFAULT_PAIR_OPERATORS = ("ts_corr", "ts_cov")


@dataclass(frozen=True)
class OperatorPool:
    """按元数和时序特征分类的不可变算子池。"""

    unary: tuple[str, ...]
    window: tuple[str, ...]
    binary: tuple[str, ...]
    pair: tuple[str, ...]

    @property
    def kinds(self) -> tuple[str, ...]:
        """返回当前至少包含一个算子的分类名称。"""
        result = []
        if self.unary:
            result.append("unary")
        if self.window:
            result.append("window")
        if self.binary:
            result.append("binary")
        if self.pair:
            result.append("pair")
        return tuple(result)

    def as_sample_kwargs(self) -> dict[str, tuple[str, ...]]:
        """转换为 Bayesian 公式采样器需要的关键字参数。"""
        return {
            "unary_operators": self.unary,
            "window_operators": self.window,
            "binary_operators": self.binary,
            "pair_operators": self.pair,
        }


DEFAULT_OPERATOR_GROUPS: dict[str, tuple[str, ...]] = {
    "unary": DEFAULT_UNARY_OPERATORS,
    "window": DEFAULT_WINDOW_OPERATORS,
    "binary": DEFAULT_BINARY_OPERATORS,
    "pair": DEFAULT_PAIR_OPERATORS,
}


def build_operator_pool(
    operator_config: Mapping[str, Sequence[str]] | None = None,
    *,
    consumer: str = "因子挖掘器",
    use_defaults: bool = True,
) -> OperatorPool:
    """校验算子配置，并与当前实际注册的算子取交集。

    用户只能从 Orion 公共默认算子集合中缩小搜索范围，不能通过配置注入任意
    函数。自由挖掘可用默认值补齐未指定分类；定向挖掘关闭默认补齐后，未指定
    分类视为空。不同部署若缺少部分算子，会跳过缺失项并发出中文警告。
    """
    config = dict(operator_config or {})
    unknown = sorted(set(config) - set(DEFAULT_OPERATOR_GROUPS))
    if unknown:
        raise KeyError(f"未知 operator_config 分类: {unknown}")

    available = formula_ops.OPERATORS
    resolved: dict[str, tuple[str, ...]] = {}
    missing: set[str] = set()
    for category, default_names in DEFAULT_OPERATOR_GROUPS.items():
        fallback = default_names if use_defaults else ()
        requested = tuple(dict.fromkeys(config.get(category, fallback)))
        unsupported = sorted(set(requested) - set(default_names))
        if unsupported:
            raise ValueError(f"{category} 包含不兼容算子: {unsupported}")
        resolved[category] = tuple(
            name for name in requested if name in available
        )
        missing.update(name for name in requested if name not in available)

    if missing:
        warnings.warn(
            f"{consumer} 已跳过 formula.py 未注册的算子: "
            + ", ".join(sorted(missing)),
            RuntimeWarning,
            stacklevel=2,
        )
    pool = OperatorPool(**resolved)
    if not pool.kinds:
        raise RuntimeError(
            f"{consumer} 没有可用算子；请检查 feature.utils.formula.OPERATORS"
        )
    return pool


__all__ = [
    "DEFAULT_BINARY_OPERATORS",
    "DEFAULT_OPERATOR_GROUPS",
    "DEFAULT_PAIR_OPERATORS",
    "DEFAULT_UNARY_OPERATORS",
    "DEFAULT_WINDOW_OPERATORS",
    "OperatorPool",
    "build_operator_pool",
]

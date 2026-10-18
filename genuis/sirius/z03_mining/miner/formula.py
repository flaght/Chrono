"""因子挖掘后端共用的公式树、算子集合与 Polars 编译器。"""

from __future__ import annotations

from typing import Any, Literal
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import polars as pl
import feature.utils.formula as ops


MiningMode = Literal["free", "directed"]


def validate_names(values: Iterable[str], argument: str) -> tuple[str, ...]:
    """去重并校验特征列名，同时保持调用方传入的原始顺序。"""
    names = tuple(dict.fromkeys(values))
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{argument} 必须包含至少一个非空列名")
    return names



@dataclass(frozen=True)
class Formula:
    """描述特征叶子或算子节点的不可变轻量公式树。"""

    operator: str
    children: tuple["Formula", ...] = ()
    feature: str | None = None
    period: int | None = None

    @classmethod
    def leaf(cls, feature: str) -> "Formula":
        """根据基础特征列名创建一个公式叶子节点。"""
        return cls("feature", feature=feature)

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 JSON 的嵌套字典。"""
        result: dict[str, Any] = {"operator": self.operator}
        if self.feature is not None:
            result["feature"] = self.feature
        if self.period is not None:
            result["period"] = self.period
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result

    @property
    def text(self) -> str:
        """返回便于阅读和记录的前缀函数表达式。"""
        if self.operator == "feature":
            return str(self.feature)
        arguments = [child.text for child in self.children]
        if self.period is not None:
            arguments.append(str(self.period))
        return f"{self.operator}({', '.join(arguments)})"

    @property
    def factor_id(self) -> str:
        """根据规范化公式结构生成稳定的短标识。"""
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def features(self) -> frozenset[str]:
        """递归收集公式依赖的全部基础特征列名。"""
        if self.operator == "feature":
            return frozenset((str(self.feature),))
        return frozenset().union(*(child.features for child in self.children))


class FormulaCompiler:
    """将公式树编译为 LazyFrame，并逐层物化中间计算节点。"""

    def __init__(self, key_columns: Sequence[str] = ("trade_time", "code")):
        """设置时间列和品种列等结果主键。"""
        self.key_columns = tuple(key_columns)

    def compile(
        self,
        df_lazy: pl.LazyFrame,
        formula: Formula,
        *,
        output_name: str = "factor",
    ) -> pl.LazyFrame:
        """编译完整公式，并只返回主键列和指定名称的因子列。"""
        # 时序算子依赖稳定顺序，因此所有公式统一先按主键排序。
        frame = df_lazy.sort(list(self.key_columns))
        frame, column = self._compile_node(frame, formula, {})
        return frame.select([*self.key_columns, pl.col(column).alias(output_name)])

    def _compile_node(
        self,
        frame: pl.LazyFrame,
        node: Formula,
        cache: dict[str, str],
    ) -> tuple[pl.LazyFrame, str]:
        """递归编译单个节点，返回更新后的数据图及该节点结果列名。"""
        # 叶子节点直接引用输入列，不产生额外中间列。
        if node.operator == "feature":
            if node.feature is None:
                raise ValueError("feature 节点缺少列名")
            return frame, node.feature
        if node.operator not in ops.OPERATORS:
            raise KeyError(f"未知算子: {node.operator}")
        # 同一子树只物化一次，后续引用直接复用缓存列。
        if node.factor_id in cache:
            return frame, cache[node.factor_id]

        # 先完成全部子节点，再把子节点结果列传给当前算子。
        child_columns: list[str] = []
        for child in node.children:
            frame, column = self._compile_node(frame, child, cache)
            child_columns.append(column)
        operator = ops.OPERATORS[node.operator]
        arguments: list[Any] = list(child_columns)
        if node.period is not None:
            arguments.append(node.period)
        expression = operator(*arguments)
        # 使用公式标识构造内部列名，降低不同子树发生重名的风险。
        column = f"_formula_{node.factor_id}"
        frame = ops.materialize(frame, expression, column)
        cache[node.factor_id] = column
        return frame, column


__all__ = [
    "Formula", "FormulaCompiler", "MiningMode", "validate_names",
]

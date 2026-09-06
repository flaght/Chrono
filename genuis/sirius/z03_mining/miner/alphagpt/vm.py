"""将后缀公式 Token 编译为 Polars 表达式的虚拟机。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from miner.formula import Formula, FormulaCompiler

from .ops import OperatorSpec


@dataclass(frozen=True)
class DecodedFormula:
    formula: Formula
    tokens: tuple[int, ...]

    @property
    def expression(self) -> str:
        return self.formula.text


class StackVM:
    """校验后缀 Token，并使用 Polars 执行解码后的公式图。"""

    def __init__(
        self,
        feature_names: Sequence[str],
        operator_specs: Sequence[OperatorSpec],
        *,
        key_columns: Sequence[str] = ("trade_time", "code"),
    ) -> None:
        self.feature_names = tuple(feature_names)
        self.operator_specs = tuple(operator_specs)
        self.n_features = len(self.feature_names)
        self.vocab = self.feature_names + tuple(
            spec.token_name for spec in self.operator_specs
        )
        self.compiler = FormulaCompiler(key_columns=key_columns)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def operator_spec(self, token: int) -> OperatorSpec | None:
        index = token - self.n_features
        if 0 <= index < len(self.operator_specs):
            return self.operator_specs[index]
        return None

    def decode(self, formula_tokens: Sequence[int]) -> DecodedFormula | None:
        stack: list[Formula] = []
        normalized_tokens = tuple(int(token) for token in formula_tokens)
        for token in normalized_tokens:
            if 0 <= token < self.n_features:
                stack.append(Formula.leaf(self.feature_names[token]))
                continue
            spec = self.operator_spec(token)
            if spec is None or len(stack) < spec.arity:
                return None
            children = tuple(stack[-spec.arity:])
            del stack[-spec.arity:]
            stack.append(
                Formula(spec.operator, children, period=spec.period)
            )
        if len(stack) != 1:
            return None
        return DecodedFormula(stack[0], normalized_tokens)

    def execute(
        self,
        formula_tokens: Sequence[int],
        df_lazy: pl.LazyFrame,
        *,
        output_name: str = "factor",
    ) -> pl.LazyFrame | None:
        decoded = self.decode(formula_tokens)
        if decoded is None:
            return None
        return self.compiler.compile(
            df_lazy, decoded.formula, output_name=output_name
        )

    def valid_token_mask(
        self, stack_depth: int, remaining_steps: int
    ) -> list[bool]:
        """只允许在剩余步数内最终能收敛为一个栈元素的 Token。"""
        result: list[bool] = []
        for token in range(self.vocab_size):
            next_depth = self.next_stack_depth(stack_depth, token)
            # 每个后续二元算子最多只能让栈深度减少一层。
            can_finish = 1 <= next_depth <= remaining_steps + 1
            result.append(can_finish)
        return result

    def next_stack_depth(self, stack_depth: int, token: int) -> int:
        if token < self.n_features:
            return stack_depth + 1
        spec = self.operator_spec(token)
        if spec is None or stack_depth < spec.arity:
            return -1
        return stack_depth - spec.arity + 1


__all__ = ["DecodedFormula", "StackVM"]

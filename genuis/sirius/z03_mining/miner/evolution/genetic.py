"""公式树的初始化、交叉和变异操作。"""

from __future__ import annotations

import random
from collections.abc import Sequence

from miner.formula import (
    Formula,
    validate_names,
)
from miner.operators import OperatorPool


class FormulaGenetics:
    """在 Formula AST 和调用方指定的特征空间内执行遗传操作。"""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        periods: Sequence[int],
        max_depth: int,
        operators: OperatorPool,
        random_state: random.Random,
    ) -> None:
        self.features = validate_names(feature_names, "feature_names")
        self.periods = tuple(dict.fromkeys(periods))
        self.max_depth = max_depth
        self.operators = operators
        self.random = random_state

    def random_formula(self) -> Formula:
        """生成随机公式，全部叶子均来自 feature_names。"""
        root = Formula.leaf(self.random.choice(self.features))
        depth = self.random.randint(1, self.max_depth)
        for _ in range(depth):
            root = self._wrap(root)
        return root

    def crossover(self, left: Formula, right: Formula) -> Formula:
        """把右亲本的一棵随机子树嫁接到左亲本。"""
        left_path = self.random.choice(self.paths(left))
        right_path = self.random.choice(self.paths(right))
        child = self.replace(left, left_path, self.node_at(right, right_path))
        return child if self._acceptable(child) else left

    def subtree_mutation(self, formula: Formula) -> Formula:
        """用随机生成的子树替换一个节点。"""
        path = self.random.choice(self.paths(formula))
        child = self.replace(formula, path, self.random_formula())
        return child if self._acceptable(child) else formula

    def point_mutation(self, formula: Formula) -> Formula:
        """保持节点元数不变，修改一个特征、算子或周期。"""
        path = self.random.choice(self.paths(formula))
        node = self.node_at(formula, path)
        if node.operator == "feature":
            replacement = Formula.leaf(self.random.choice(self.features))
        elif node.operator in self.operators.unary:
            replacement = Formula(
                self.random.choice(self.operators.unary), node.children
            )
        elif node.operator in self.operators.window:
            replacement = Formula(
                self.random.choice(self.operators.window), node.children,
                period=self.random.choice(self.periods),
            )
        elif node.operator in self.operators.binary:
            replacement = Formula(
                self.random.choice(self.operators.binary), node.children
            )
        elif node.operator in self.operators.pair:
            replacement = Formula(
                self.random.choice(self.operators.pair), node.children,
                period=self.random.choice(self.periods),
            )
        else:
            return formula
        child = self.replace(formula, path, replacement)
        return child if self._acceptable(child) else formula

    def hoist_mutation(self, formula: Formula) -> Formula:
        """提升一棵随机子树作为新根，以抑制公式树膨胀。"""
        candidates = self.paths(formula)[1:]
        if not candidates:
            return formula
        child = self.node_at(formula, self.random.choice(candidates))
        return child if self._acceptable(child) else formula

    def _wrap(self, root: Formula) -> Formula:
        kind = self.random.choice(self.operators.kinds)
        if kind == "unary":
            return Formula(self.random.choice(self.operators.unary), (root,))
        if kind == "window":
            return Formula(
                self.random.choice(self.operators.window), (root,),
                period=self.random.choice(self.periods),
            )
        right = Formula.leaf(self.random.choice(self.features))
        if kind == "binary":
            return Formula(
                self.random.choice(self.operators.binary), (root, right)
            )
        return Formula(
            self.random.choice(self.operators.pair), (root, right),
            period=self.random.choice(self.periods),
        )

    def _acceptable(self, formula: Formula) -> bool:
        return self.depth(formula) <= self.max_depth

    @classmethod
    def depth(cls, formula: Formula) -> int:
        if not formula.children:
            return 0
        return 1 + max(cls.depth(child) for child in formula.children)

    @classmethod
    def paths(cls, formula: Formula) -> list[tuple[int, ...]]:
        result = [()]
        for index, child in enumerate(formula.children):
            result.extend(
                (index, *child_path) for child_path in cls.paths(child)
            )
        return result

    @staticmethod
    def node_at(formula: Formula, path: tuple[int, ...]) -> Formula:
        node = formula
        for index in path:
            node = node.children[index]
        return node

    @classmethod
    def replace(
        cls,
        formula: Formula,
        path: tuple[int, ...],
        replacement: Formula,
    ) -> Formula:
        if not path:
            return replacement
        index = path[0]
        children = list(formula.children)
        children[index] = cls.replace(children[index], path[1:], replacement)
        return Formula(
            formula.operator,
            tuple(children),
            feature=formula.feature,
            period=formula.period,
        )


__all__ = ["FormulaGenetics"]

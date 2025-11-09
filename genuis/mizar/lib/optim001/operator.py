import optuna, re
from typing import List, Dict, Any
from lib.optim001.parser import FactorExpressionParser
from ultron.factor.genetic.geneticist.operators import *


class OperatorReplacer:
    """算子替换器"""

    def __init__(self, parser: FactorExpressionParser):
        self.parser = parser

    def get_replacement_candidates(self, operator: str) -> List[str]:
        """
        获取算子的替换候选
        
        Args:
            operator: 原始算子名称
            
        Returns:
            替换候选列表
        """
        if operator not in self.parser.operators:
            return []

        original_op = self.parser.operators[operator]
        candidates = []

        # 根据算子类别和参数个数寻找相似算子
        for op_name, op_info in self.parser.operators.items():
            if (op_info.category == original_op.category
                    and op_info.param_count == original_op.param_count
                    and op_name != operator):
                candidates.append(op_name)

        return candidates

    def suggest_operator_replacements(self, trial: optuna.Trial,
                                      expression: str) -> Dict[str, str]:
        """
        为表达式中的算子提供替换建议
        
        Args:
            trial: Optuna试验对象
            expression: 因子表达式
            
        Returns:
            算子替换建议字典
        """
        parsed = self.parser.parse_expression(expression)
        replacements = {}

        for operator in parsed['operators']:
            candidates = self.get_replacement_candidates(operator)
            if candidates:
                # 决定是否替换这个算子
                should_replace = trial.suggest_categorical(
                    f'replace_{operator}', [True, False])
                if should_replace:
                    new_operator = trial.suggest_categorical(
                        f'new_{operator}', candidates)
                    replacements[operator] = new_operator

        return replacements

    def replace_operators(self, expression: str,
                          replacements: Dict[str, str]) -> str:
        """
        用新的算子替换表达式中的算子
        
        Args:
            expression: 原始表达式
            replacements: 算子替换字典
            
        Returns:
            替换后的表达式
        """
        result = expression

        for old_op, new_op in replacements.items():
            # 使用词边界确保完整匹配
            pattern = r'\b' + re.escape(old_op) + r'\b'
            result = re.sub(pattern, new_op, result)

        return result

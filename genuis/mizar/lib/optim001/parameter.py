from typing import List, Dict, Any
import optuna
from lib.optim001.parser import FactorExpressionParser
from ultron.factor.genetic.geneticist.operators import *

class ParameterOptimizer:
    """参数优化器"""

    def __init__(self, parser: FactorExpressionParser,):
        self.parser = parser

    def suggest_parameters(self, trial: optuna.Trial,
                           expression: str) -> Dict[str, Any]:
        """
        为表达式中的参数提供建议值
        
        Args:
            trial: Optuna试验对象
            expression: 因子表达式
            
        Returns:
            参数建议字典
        """
        parsed = self.parser.parse_expression(expression)
        suggestions = {}

        for param in parsed['parameters']:
            # 为不同的参数类型提供不同的建议范围
            if param <= 5:  # 短期参数
                suggestions[f'param_{param}'] = trial.suggest_int(
                    f'param_{param}', 3, 20)
            elif param <= 20:  # 中期参数
                suggestions[f'param_{param}'] = trial.suggest_int(
                    f'param_{param}', 10, 60)
            else:  # 长期参数
                suggestions[f'param_{param}'] = trial.suggest_int(
                    f'param_{param}', 30, 120)

        return suggestions

    def replace_parameters(self, expression: str,
                           suggestions: Dict[str, Any]) -> str:
        """
        用建议的参数值替换表达式中的参数
        
        Args:
            expression: 原始表达式
            suggestions: 参数建议字典，格式为 {'param_20': 54, 'param_5': 7}
            
        Returns:
            替换后的表达式
        """
        result = expression
        # 从suggestions字典中提取原参数值和新参数值的映射
        # 例如: {'param_20': 54} -> {20: 54}
        param_mapping = {}
        for key, new_value in suggestions.items():
            if key.startswith('param_'):
                old_value = int(key.replace('param_', ''))
                param_mapping[old_value] = new_value
        
        # 按原参数值降序排序，避免替换冲突
        # 例如：先替换20，再替换5，避免将20中的2误替换
        sorted_params = sorted(param_mapping.items(), key=lambda x: x[0], reverse=True)
        
        for old_param, new_param in sorted_params:
            # 使用更精确的替换方式，确保只替换数字参数
            # 避免替换字符串中的数字或其他非参数位置的数字
            import re
            # 匹配函数调用中的第一个参数（数字），例如 MA(20, ...) 中的 20
            pattern = r'\b' + str(old_param) + r'\b'
            result = re.sub(pattern, str(new_param), result, count=1)
        
        return result
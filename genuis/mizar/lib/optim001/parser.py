import re, pdb
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any
from ultron.factor.genetic.geneticist.operators import *


@dataclass
class OperatorInfo:
    """算子信息"""
    name: str
    category: str
    description: str
    param_count: int  # 参数个数
    is_time_series: bool  # 是否为时序算子


@dataclass
class FieldInfo:
    """字段信息"""
    name: str
    type: str
    description: str
    dependencies: List[str]


class FactorExpressionParser:
    """因子表达式解析器"""

    def __init__(self, operators_pd: pd.DataFrame, fields_pd: pd.DataFrame):
        self.fields = self._init_fields(fields_pd=fields_pd)
        self.operators = self._init_operators(operators_pd=operators_pd)

    def _count_parameters(self, expression):
        """从表达式中计算参数个数"""
        # 简单的参数计数逻辑
        if '(' not in expression:
            return 0

        # 提取括号内的内容
        match = re.search(r'\(([^)]*)\)', expression)
        if not match:
            return 0

        params_str = match.group(1)
        if not params_str.strip():
            return 0

        # 计算逗号分隔的参数个数
        return len([p.strip() for p in params_str.split(',') if p.strip()])

    def _parse_dependencies(self, dep_value):
        """安全解析依赖字段，支持空值/逗号分隔/带引号的项。"""
        if pd.isna(dep_value):
            return []
        text = str(dep_value).strip()
        if not text:
            return []
        # 移除包裹的引号，并按逗号切分
        items = [item.strip().strip('"\'') for item in text.split(',')]
        # 过滤空项
        return [itm for itm in items if itm]

    def _init_operators(self, operators_pd):
        operators = {}
        for row in operators_pd.itertuples():
            operator_name = row.operator_name
            category = row.category
            description = row.description

            expression = row.expression
            param_count = self._count_parameters(expression)

            is_time_series = 'window' in expression.lower(
            ) or category == '时序算子'

            operators[operator_name] = OperatorInfo(
                name=operator_name,
                category=category,
                description=description,
                param_count=param_count,
                is_time_series=is_time_series)

        return operators

    def _init_fields(self, fields_pd):
        fields = {}
        for row in fields_pd.itertuples():
            field_name = row.field_name
            field_type = row.field_type
            description = row.description
            dependencies = ''

            fields[field_name] = FieldInfo(name=field_name,
                                           type=field_type,
                                           description=description,
                                           dependencies=dependencies)

        return fields

    def parse_expression(self, expression: str) -> Dict[str, Any]:
        """
        解析因子表达式
        
        Args:
            expression: 因子表达式字符串
            
        Returns:
            解析结果字典
        """
        result = {
            'original_expression': expression,
            'operators': [],
            'parameters': [],
            'fields': [],
            'structure': {}
        }

        # 提取所有算子
        operator_pattern = r'\b([A-Z][A-Z0-9_]*)\s*\('
        operators_found = re.findall(operator_pattern, expression)
        result['operators'] = list(set(operators_found))

        # 提取所有字段
        ## 转表达式
        #field_pattern = r'\b(open|close|high|low|volume|money|twap|pct_change|pct_change_close|pct_change_set)\b'
        #fields_found = re.findall(field_pattern, expression)
        result['fields'] = list(set(eval(expression)._dependency))

        # 提取数值参数
        param_pattern = r'\b(\d+)\b'
        params_found = re.findall(param_pattern, expression)
        result['parameters'] = [int(p) for p in params_found]

        # 分析表达式结构
        result['structure'] = self._analyze_structure(expression)

        return result

    def _analyze_structure(self, expression: str) -> Dict[str, Any]:
        """分析表达式结构"""
        structure = {
            'depth': 0,
            'nested_operators': [],
            'parameter_positions': []
        }

        # 计算嵌套深度
        depth = 0
        max_depth = 0
        for char in expression:
            if char == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ')':
                depth -= 1

        structure['depth'] = max_depth

        # 找到嵌套的算子
        nested_pattern = r'([A-Z][A-Z0-9_]*)\s*\([^)]*\([^)]*\)[^)]*\)'
        nested_ops = re.findall(nested_pattern, expression)
        structure['nested_operators'] = list(set(nested_ops))

        return structure
import re, pdb
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any
from ultron.factor.genetic.geneticist.operators import *


@dataclass
class OperatorInfo:
    """算子信息"""
    name: str
    category: str
    description: str
    param_count: int  # 参数个数
    is_time_series: bool  # 是否为时序算子


@dataclass
class FieldInfo:
    """字段信息"""
    name: str
    type: str
    description: str
    dependencies: List[str]


class FactorExpressionParser:
    """因子表达式解析器"""

    def __init__(self, operators_pd: pd.DataFrame, fields_pd: pd.DataFrame):
        self.fields = self._init_fields(fields_pd=fields_pd)
        self.operators = self._init_operators(operators_pd=operators_pd)

    def _count_parameters(self, expression):
        """从表达式中计算参数个数"""
        # 简单的参数计数逻辑
        if '(' not in expression:
            return 0

        # 提取括号内的内容
        match = re.search(r'\(([^)]*)\)', expression)
        if not match:
            return 0

        params_str = match.group(1)
        if not params_str.strip():
            return 0

        # 计算逗号分隔的参数个数
        return len([p.strip() for p in params_str.split(',') if p.strip()])

    def _parse_dependencies(self, dep_value):
        """安全解析依赖字段，支持空值/逗号分隔/带引号的项。"""
        if pd.isna(dep_value):
            return []
        text = str(dep_value).strip()
        if not text:
            return []
        # 移除包裹的引号，并按逗号切分
        items = [item.strip().strip('"\'') for item in text.split(',')]
        # 过滤空项
        return [itm for itm in items if itm]

    def _init_operators(self, operators_pd):
        operators = {}
        for row in operators_pd.itertuples():
            operator_name = row.operator_name
            category = row.category
            description = row.description

            expression = row.expression
            param_count = self._count_parameters(expression)

            is_time_series = 'window' in expression.lower(
            ) or category == '时序算子'

            operators[operator_name] = OperatorInfo(
                name=operator_name,
                category=category,
                description=description,
                param_count=param_count,
                is_time_series=is_time_series)

        return operators

    def _init_fields(self, fields_pd):
        fields = {}
        for row in fields_pd.itertuples():
            field_name = row.field_name
            field_type = row.field_type
            description = row.description
            dependencies = ''

            fields[field_name] = FieldInfo(name=field_name,
                                           type=field_type,
                                           description=description,
                                           dependencies=dependencies)

        return fields

    def parse_expression(self, expression: str) -> Dict[str, Any]:
        """
        解析因子表达式
        
        Args:
            expression: 因子表达式字符串
            
        Returns:
            解析结果字典
        """
        result = {
            'original_expression': expression,
            'operators': [],
            'parameters': [],
            'fields': [],
            'structure': {}
        }

        # 提取所有算子
        operator_pattern = r'\b([A-Z][A-Z0-9_]*)\s*\('
        operators_found = re.findall(operator_pattern, expression)
        result['operators'] = list(set(operators_found))

        # 提取所有字段
        ## 转表达式
        #field_pattern = r'\b(open|close|high|low|volume|money|twap|pct_change|pct_change_close|pct_change_set)\b'
        #fields_found = re.findall(field_pattern, expression)
        result['fields'] = list(set(eval(expression)._dependency))

        # 提取数值参数
        param_pattern = r'\b(\d+)\b'
        params_found = re.findall(param_pattern, expression)
        result['parameters'] = [int(p) for p in params_found]

        # 分析表达式结构
        result['structure'] = self._analyze_structure(expression)

        return result

    def _analyze_structure(self, expression: str) -> Dict[str, Any]:
        """分析表达式结构"""
        structure = {
            'depth': 0,
            'nested_operators': [],
            'parameter_positions': []
        }

        # 计算嵌套深度
        depth = 0
        max_depth = 0
        for char in expression:
            if char == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ')':
                depth -= 1

        structure['depth'] = max_depth

        # 找到嵌套的算子
        nested_pattern = r'([A-Z][A-Z0-9_]*)\s*\([^)]*\([^)]*\)[^)]*\)'
        nested_ops = re.findall(nested_pattern, expression)
        structure['nested_operators'] = list(set(nested_ops))

        return structure

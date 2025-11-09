"""
特征优化器模块
支持在因子表达式中替换和优化特征（字段）
"""

import re
from typing import Dict, List, Any, Optional
from lib.optim001.parser import FactorExpressionParser


class FieldOptimizer:
    """特征优化器"""

    def __init__(self, parser: FactorExpressionParser):
        """
        初始化特征优化器
        
        Args:
            parser: 因子表达式解析器
        """
        self.parser = parser

    def get_field_candidates(self, field: str) -> List[str]:
        """
        获取字段的替换候选
        
        Args:
            field: 原始字段名称
            
        Returns:
            替换候选列表
        """
        if field not in self.parser.fields:
            return []

        original_field = self.parser.fields[field]
        candidates = []

        # 根据字段类型寻找相似字段
        for field_name, field_info in self.parser.fields.items():
            if field_name == field:
                continue  # 跳过自身

            # 同类型字段可以互相替换
            if field_info.type == original_field.type:
                candidates.append(field_name)

        return candidates

    def suggest_field_replacements(self, trial, expression: str) -> Dict[str, str]:
        """
        使用Optuna trial建议字段替换
        
        Args:
            trial: Optuna trial对象
            expression: 当前表达式
            
        Returns:
            字段替换建议字典 {原字段: 新字段}
        """
        # 解析表达式中的字段
        parsed = self.parser.parse_expression(expression)
        fields_in_expr = parsed['fields']

        replacements = {}

        for field in fields_in_expr:
            candidates = self.get_field_candidates(field)

            if candidates:
                # 添加 None 选项（表示不替换）
                choices = [None] + candidates

                # 使用trial建议替换
                suggested = trial.suggest_categorical(f'new_field_{field}',
                                                      choices)

                if suggested is not None:
                    replacements[field] = suggested

        return replacements

    def replace_fields(self, expression: str,
                       replacements: Dict[str, str]) -> str:
        """
        在表达式中替换字段
        
        Args:
            expression: 原始表达式
            replacements: 字段替换字典 {原字段: 新字段}
            
        Returns:
            替换后的表达式
        """
        result = expression

        # 按字段名长度降序排序，避免替换冲突
        # 例如：先替换 'pct_change_close'，再替换 'close'
        sorted_fields = sorted(replacements.items(),
                              key=lambda x: len(x[0]),
                              reverse=True)

        for old_field, new_field in sorted_fields:
            # 使用正则表达式精确替换字段
            # 匹配引号内的字段名：'field_name'
            pattern = r"(['\"])" + re.escape(old_field) + r"\1"
            replacement = r"\1" + new_field + r"\1"
            result = re.sub(pattern, replacement, result)

        return result

    def suggest_field_additions(self, trial, expression: str,
                               max_additions: int = 2) -> List[str]:
        """
        建议添加新的字段到表达式
        
        Args:
            trial: Optuna trial对象
            expression: 当前表达式
            max_additions: 最多添加的字段数量
            
        Returns:
            建议添加的字段列表
        """
        parsed = self.parser.parse_expression(expression)
        fields_in_expr = set(parsed['fields'])

        # 获取所有可用字段
        all_fields = set(self.parser.fields.keys())

        # 未使用的字段
        unused_fields = list(all_fields - fields_in_expr)

        if not unused_fields:
            return []

        additions = []
        for i in range(max_additions):
            # 是否添加字段
            should_add = trial.suggest_categorical(f'add_field_{i}',
                                                   [True, False])

            if should_add and unused_fields:
                # 选择要添加的字段
                field_to_add = trial.suggest_categorical(
                    f'field_to_add_{i}', unused_fields)

                additions.append(field_to_add)
                unused_fields.remove(field_to_add)

        return additions

    def get_field_groups(self) -> Dict[str, List[str]]:
        """
        获取字段分组（按类型）
        
        Returns:
            字段分组字典 {类型: [字段列表]}
        """
        groups = {}

        for field_name, field_info in self.parser.fields.items():
            field_type = field_info.type
            if field_type not in groups:
                groups[field_type] = []
            groups[field_type].append(field_name)

        return groups

    def suggest_field_by_type(self, trial, field_type: str,
                             exclude: List[str] = None) -> Optional[str]:
        """
        按类型建议字段
        
        Args:
            trial: Optuna trial对象
            field_type: 字段类型
            exclude: 要排除的字段列表
            
        Returns:
            建议的字段名称
        """
        exclude = exclude or []

        # 获取该类型的所有字段
        candidates = [
            field_name for field_name, field_info in self.parser.fields.items()
            if field_info.type == field_type and field_name not in exclude
        ]

        if not candidates:
            return None

        # 添加 None 选项
        choices = [None] + candidates

        return trial.suggest_categorical(f'field_type_{field_type}', choices)

    def extract_field_positions(self, expression: str) -> List[Dict[str, Any]]:
        """
        提取表达式中所有字段的位置信息
        
        Args:
            expression: 因子表达式
            
        Returns:
            字段位置信息列表 [{field: str, start: int, end: int, quoted: bool}]
        """
        positions = []

        # 匹配引号内的字段：'field_name' 或 "field_name"
        pattern = r"(['\"])([a-zA-Z_][a-zA-Z0-9_]*)\1"

        for match in re.finditer(pattern, expression):
            field_name = match.group(2)

            # 检查是否为已知字段
            if field_name in self.parser.fields:
                positions.append({
                    'field': field_name,
                    'start': match.start(),
                    'end': match.end(),
                    'quoted': True,
                    'quote_char': match.group(1)
                })

        return positions

    def replace_field_at_position(self, expression: str, field_name: str,
                                  new_field: str,
                                  occurrence: int = 1) -> str:
        """
        在指定位置替换字段（按出现次数）
        
        Args:
            expression: 原始表达式
            field_name: 要替换的字段名
            new_field: 新字段名
            occurrence: 替换第几次出现（从1开始）
            
        Returns:
            替换后的表达式
        """
        # 匹配引号内的字段
        pattern = r"(['\"])" + re.escape(field_name) + r"\1"
        replacement = r"\1" + new_field + r"\1"

        # 只替换指定的第N次出现
        count = 0

        def replace_func(match):
            nonlocal count
            count += 1
            if count == occurrence:
                return match.group(1) + new_field + match.group(1)
            return match.group(0)

        result = re.sub(pattern, replace_func, expression)
        return result

    def suggest_field_replacements_by_position(
            self, trial, expression: str) -> Dict[str, Any]:
        """
        按位置建议字段替换（支持同一字段在不同位置替换为不同值）
        
        Args:
            trial: Optuna trial对象
            expression: 当前表达式
            
        Returns:
            位置替换建议字典 {position_id: {field, new_field, occurrence}}
        """
        positions = self.extract_field_positions(expression)

        # 统计每个字段的出现次数
        field_counts = {}
        for pos in positions:
            field = pos['field']
            field_counts[field] = field_counts.get(field, 0) + 1

        replacements = {}

        # 为每个位置建议替换
        for idx, pos in enumerate(positions):
            field = pos['field']
            occurrence = sum(1 for p in positions[:idx + 1]
                           if p['field'] == field)

            candidates = self.get_field_candidates(field)

            if candidates:
                choices = [None] + candidates

                # 使用唯一的参数名
                param_name = f'field_pos_{idx}_{field}_occ{occurrence}'
                suggested = trial.suggest_categorical(param_name, choices)

                if suggested is not None:
                    replacements[f'pos_{idx}'] = {
                        'field': field,
                        'new_field': suggested,
                        'occurrence': occurrence
                    }

        return replacements

    def apply_positional_replacements(
            self, expression: str,
            replacements: Dict[str, Any]) -> str:
        """
        应用基于位置的字段替换
        
        Args:
            expression: 原始表达式
            replacements: 位置替换字典
            
        Returns:
            替换后的表达式
        """
        result = expression

        # 按位置索引排序（从后往前替换，避免位置偏移）
        sorted_replacements = sorted(replacements.items(),
                                    key=lambda x: int(x[0].split('_')[1]),
                                    reverse=True)

        for pos_id, replacement in sorted_replacements:
            field = replacement['field']
            new_field = replacement['new_field']
            occurrence = replacement['occurrence']

            result = self.replace_field_at_position(result, field, new_field,
                                                   occurrence)

        return result


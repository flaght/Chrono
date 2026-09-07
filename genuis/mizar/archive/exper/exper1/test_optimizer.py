import os, re, pdb, random
import optuna
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Union


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

    def __init__(self, operators_csv_path: str, fields_csv_path: str):
        """
        初始化解析器
        
        Args:
            operators_csv_path: 算子依赖关系CSV文件路径
            fields_csv_path: 字段依赖关系CSV文件路径
        """
        self.operators = self._load_operators(operators_csv_path)
        self.fields = self._load_fields(fields_csv_path)

    def _load_operators(self, csv_path: str) -> Dict[str, OperatorInfo]:
        """加载算子信息"""
        df = pd.read_csv(csv_path)
        operators = {}

        for _, row in df.iterrows():
            operator_name = row['Operator']
            category = row['Category']
            description = row['Description']

            # 从表达式中推断参数个数
            expression = row['Expression']
            param_count = self._count_parameters(expression)

            # 判断是否为时序算子
            is_time_series = 'window' in expression.lower(
            ) or category == '时序算子'

            operators[operator_name] = OperatorInfo(
                name=operator_name,
                category=category,
                description=description,
                param_count=param_count,
                is_time_series=is_time_series)

        return operators

    def _load_fields(self, csv_path: str) -> Dict[str, FieldInfo]:
        """加载字段信息"""
        df = pd.read_csv(csv_path)
        fields = {}

        def _parse_dependencies(dep_value: Any) -> List[str]:
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

        for _, row in df.iterrows():
            field_name = row['Field']
            field_type = row['types']
            description = row['Description']
            dependencies = _parse_dependencies(row.get('Dependencies', ''))

            fields[field_name] = FieldInfo(name=field_name,
                                           type=field_type,
                                           description=description,
                                           dependencies=dependencies)

        return fields

    def _count_parameters(self, expression: str) -> int:
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
        pdb.set_trace()
        # 提取所有字段
        field_pattern = r'\b(open|close|high|low|volume|money|twap|pct_change|pct_change_close|pct_change_set)\b'
        fields_found = re.findall(field_pattern, expression)
        result['fields'] = list(set(fields_found))

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


class ParameterOptimizer:
    """参数优化器"""

    def __init__(self, parser: FactorExpressionParser):
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
            suggestions: 参数建议字典
            
        Returns:
            替换后的表达式
        """
        result = expression
        param_values = list(suggestions.values())

        # 按参数值大小排序，确保替换顺序正确
        sorted_params = sorted(param_values, reverse=True)

        for i, param in enumerate(sorted_params):
            # 替换第一个匹配的参数
            result = result.replace(str(param), str(param_values[i]), 1)

        return result


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


class FactorOptimizer:
    """因子优化器主类"""

    def __init__(self, operators_csv_path: str, fields_csv_path: str):
        """
        初始化因子优化器
        
        Args:
            operators_csv_path: 算子依赖关系CSV文件路径
            fields_csv_path: 字段依赖关系CSV文件路径
        """
        self.parser = FactorExpressionParser(operators_csv_path,
                                             fields_csv_path)
        self.param_optimizer = ParameterOptimizer(self.parser)
        self.op_replacer = OperatorReplacer(self.parser)

    def optimize_expression(
            self,
            expression: str,
            objective_function: callable,
            n_trials: int = 100,
            optimize_parameters: bool = True,
            optimize_operators: bool = True,
            study_name: str = "factor_optimization") -> Dict[str, Any]:
        """
        优化因子表达式
        
        Args:
            expression: 原始因子表达式
            objective_function: 目标函数，接受优化后的表达式作为参数
            n_trials: 优化试验次数
            optimize_parameters: 是否优化参数
            optimize_operators: 是否优化算子
            study_name: 研究名称
            
        Returns:
            优化结果字典
        """

        def objective(trial):
            """Optuna目标函数"""
            current_expression = expression

            # 参数优化
            if optimize_parameters:
                param_suggestions = self.param_optimizer.suggest_parameters(
                    trial, current_expression)
                current_expression = self.param_optimizer.replace_parameters(
                    current_expression, param_suggestions)

            # 算子替换
            if optimize_operators:
                op_replacements = self.op_replacer.suggest_operator_replacements(
                    trial, current_expression)
                current_expression = self.op_replacer.replace_operators(
                    current_expression, op_replacements)

            # 计算目标函数值
            try:
                score = objective_function(current_expression)
                return score
            except Exception as e:
                logger.warning(f"目标函数计算失败: {e}")
                return float('-inf')

        # 创建Optuna研究（将数据库保存到模块同级的 temp 目录）
        base_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(base_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        db_path = os.path.join(temp_dir, f'{study_name}.db')
        
        study = optuna.create_study(direction='maximize',
                                    study_name=study_name,
                                    storage=f'sqlite:///{db_path}',
                                    load_if_exists=True)

        # 运行优化
        study.optimize(objective, n_trials=n_trials)

        # 返回优化结果
        best_trial = study.best_trial
        best_expression = self._reconstruct_best_expression(
            expression, best_trial.params, optimize_parameters,
            optimize_operators)

        return {
            'best_score': best_trial.value,
            'best_expression': best_expression,
            'best_params': best_trial.params,
            'study': study,
            'n_trials': len(study.trials)
        }

    def _reconstruct_best_expression(self, original_expression: str,
                                     best_params: Dict[str, Any],
                                     optimize_parameters: bool,
                                     optimize_operators: bool) -> str:
        """重构最佳表达式"""
        expression = original_expression

        # 重构参数
        if optimize_parameters:
            param_suggestions = {
                k: v
                for k, v in best_params.items() if k.startswith('param_')
            }
            expression = self.param_optimizer.replace_parameters(
                expression, param_suggestions)

        # 重构算子
        if optimize_operators:
            op_replacements = {}
            for k, v in best_params.items():
                if k.startswith('new_') and v is not None:
                    old_op = k.replace('new_', '')
                    op_replacements[old_op] = v
            expression = self.op_replacer.replace_operators(
                expression, op_replacements)

        return expression


def test_parameter_optimization1():
    """测试参数优化"""
    print("=" * 50)
    print("测试参数优化")
    print("=" * 50)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    operators_csv = os.path.join(base_dir, "resource",
                                 "expression_dependencies.csv")
    fields_csv = os.path.join(base_dir, "resource",
                              "level2_fields_dependencies.csv")

    optimizer = FactorOptimizer(operators_csv, fields_csv)

    # 简单目标函数
    def simple_objective(expression: str) -> float:
        # 基于表达式长度和参数数量的简单评分
        base_score = 0.5
        param_count = len([c for c in expression if c.isdigit()])
        length_penalty = len(expression) * 0.001
        return base_score + param_count * 0.1 - length_penalty

    # 测试表达式
    test_expr = "MA(20, close)"

    print(f"原始表达式: {test_expr}")
    print(f"原始分数: {simple_objective(test_expr):.4f}")

    pdb.set_trace()
    # 运行优化（少量试验）
    result = optimizer.optimize_expression(
        expression=test_expr,
        objective_function=simple_objective,
        n_trials=10,
        optimize_parameters=True,
        optimize_operators=False,
        study_name="test_param_optimization")

    print(f"优化后分数: {result['best_score']:.4f}")
    print(f"优化后表达式: {result['best_expression']}")
    print()

    return True


def test_parameter_optimization2():
    print("=" * 50)
    print("测试RPV因子优化")
    print("=" * 50)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    operators_csv = os.path.join(base_dir, "resource",
                                 "expression_dependencies.csv")
    fields_csv = os.path.join(base_dir, "resource",
                              "level2_fields_dependencies.csv")

    optimizer = FactorOptimizer(operators_csv, fields_csv)

    # RPV因子目标函数
    def rpv_objective(expression: str) -> float:
        score = 0.2
        '''
        # 检查是否包含RPV关键组件
        if 'MCORR' in expression:
            score += 0.2
        if 'SUBBED' in expression:
            score += 0.1
        if 'DIV' in expression:
            score += 0.1
        if 'MSTD' in expression:
            score += 0.1
        if 'AVG' in expression:
            score += 0.1

        # 检查参数合理性
        import re
        params = re.findall(r'\b(\d+)\b', expression)
        for param in params:
            param_val = int(param)
            if 10 <= param_val <= 30:
                score += 0.05
        '''

        #return min(1.0, score)
        return random.random()
    
    # 简化的RPV表达式
    rpv_expr = "ADDED(MCORR(20, close, volume), AVG(MCORR(20, close, volume)))"
    pdb.set_trace()
    print(f"原始RPV表达式: {rpv_expr}")
    print(f"原始分数: {rpv_objective(rpv_expr):.4f}")

    result = optimizer.optimize_expression(
        expression=rpv_expr,
        objective_function=rpv_objective,
        n_trials=15,
        optimize_parameters=True,
        optimize_operators=True,
        study_name="test_rpv_optimization"
    )
    pdb.set_trace()
    print(f"优化后分数: {result['best_score']:.4f}")
    print(f"优化后表达式: {result['best_expression']}")
    print(f"最佳参数: {result['best_params']}")
    print()

    return True


test_parameter_optimization2()

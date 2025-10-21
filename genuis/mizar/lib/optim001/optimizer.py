import pandas as pd
import optuna, pdb
from typing import List, Dict, Any
from ultron.factor.genetic.geneticist.operators import *
from lib.optim001.parser import FactorExpressionParser
from lib.optim001.parameter import ParameterOptimizer
from lib.optim001.operator import OperatorReplacer
from lib.optim001.field import FieldOptimizer


class FactorOptimizer:
    """因子优化器主类"""

    def __init__(self, operators_pd: pd.DataFrame, fields_pd: pd.DataFrame):
        """
        初始化因子优化器
        
        """
        self.parser = FactorExpressionParser(operators_pd=operators_pd,
                                             fields_pd=fields_pd)
        self.param_optimizer = ParameterOptimizer(self.parser)
        self.op_replacer = OperatorReplacer(self.parser)
        self.field_optimizer = FieldOptimizer(self.parser)

    def optimize_expression(self,
                            expression: str,
                            objective_function: callable,
                            total_data: pd.DataFrame,
                            total_data1: pd.DataFrame,
                            period: int,
                            optimize_rule: Dict,
                            n_trials: int = 100,
                            optimize_parameters: bool = True,
                            optimize_operators: bool = True,
                            optimize_fields: bool = True,
                            study_name: str = "factor_optimization",
                            multi_objective: bool = False,
                            top_n: int = 1) -> Dict[str, Any]:
        """
        优化因子表达式
        
        Args:
            expression: 原始因子表达式
            objective_function: 目标函数，接受优化后的表达式作为参数
            n_trials: 优化试验次数
            optimize_parameters: 是否优化参数
            optimize_operators: 是否优化算子
            optimize_fields: 是否优化特征（字段）
            study_name: 研究名称
            multi_objective: 是否使用多目标优化
            top_n: 返回前N个最佳结果（默认为1）
            
        Returns:
            优化结果字典，包含top_n个最佳结果
        """

        def objective(trial, expression: str, period: int,
                      total_data: pd.DataFrame, total_data1: pd.DataFrame,
                      optimize_rule: Dict):
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

            # 特征优化
            if optimize_fields:
                field_replacements = self.field_optimizer.suggest_field_replacements(
                    trial, current_expression)
                current_expression = self.field_optimizer.replace_fields(
                    current_expression, field_replacements)

            # 计算目标函数值
            try:
                result = objective_function(expression=current_expression,
                                            period=period,
                                            total_data=total_data,
                                            total_data1=total_data1,
                                            optimize_rule=optimize_rule)

                if multi_objective:
                    # 多目标优化：返回元组
                    if isinstance(result, tuple) or isinstance(result, list):
                        return result
                    else:
                        # 如果目标函数没有返回正确的多目标格式，返回默认值
                        return (0.0, 0.0, 0.0)
                else:
                    # 单目标优化：返回标量
                    return result

            except Exception as e:
                print(f"目标函数计算失败: {e}")
                if multi_objective:
                    return (0.0, 0.0, 0.0)
                else:
                    return float('-inf')

        # 创建Optuna研究（将数据库保存到模块同级的 temp 目录）
        #base_dir = os.path.dirname(os.path.abspath(__file__))
        #temp_dir = os.path.join(base_dir, 'temp')
        #os.makedirs(temp_dir, exist_ok=True)
        #db_path = os.path.join(temp_dir, f'{study_name}.db')

        if multi_objective:
            # 多目标优化：使用NSGA-II算法
            study = optuna.create_study(
                directions=list(optimize_rule.values()),
                study_name=study_name,
                sampler=optuna.samplers.NSGAIISampler())
        else:
            # 单目标优化
            study = optuna.create_study(direction='maximize',
                                        study_name=study_name)
        # 运行优化
        study.optimize(lambda trial: objective(trial=trial,
                                               expression=expression,
                                               period=period,
                                               total_data=total_data,
                                               total_data1=total_data1,
                                               optimize_rule=optimize_rule),
                       n_trials=n_trials)
        
        # 保存优化配置到结果中
        optimize_config = {
            'optimize_parameters': optimize_parameters,
            'optimize_operators': optimize_operators,
            'optimize_fields': optimize_fields
        }
        # 返回优化结果
        top_trials = self._get_top_n_trials(study, top_n, multi_objective)

        if multi_objective:
            # 多目标优化结果处理
            if top_trials:
                # 构建topN结果列表
                top_results = []
                for i, trial in enumerate(top_trials):
                    trial_expression = self._reconstruct_expression_from_trial(
                        expression, trial, optimize_parameters,
                        optimize_operators, optimize_fields)

                    top_results.append({
                        'rank': i + 1,
                        'score': tuple(trial.values),  # 多目标值，转换为元组
                        'expression': trial_expression,
                        'params': trial.params,
                        'trial': trial
                    })

                # 保持向后兼容性：best_* 字段仍然指向第一个（最佳）结果
                best_result = top_results[0] if top_results else None

                return {
                    'best_score':
                    best_result['score'] if best_result else (0.0, 0.0, 0.0),
                    'best_expression':
                    best_result['expression'] if best_result else expression,
                    'best_params':
                    best_result['params'] if best_result else {},
                    'study':
                    study,
                    'n_trials':
                    len(study.trials),
                    'pareto_front':
                    study.best_trials,  # 完整的帕累托前沿
                    'top_n_results':
                    top_results,  # 前N个结果
                    'top_n':
                    top_n
                }
            else:
                return {
                    'best_score': (0.0, 0.0, 0.0),
                    'best_expression': expression,
                    'best_params': {},
                    'study': study,
                    'n_trials': len(study.trials),
                    'pareto_front': [],
                    'top_n_results': [],
                    'top_n': top_n
                }
        else:
            # 单目标优化结果处理
            if top_trials:
                # 构建topN结果列表
                top_results = []
                for i, trial in enumerate(top_trials):
                    trial_expression = self._reconstruct_expression_from_trial(
                        expression, trial, optimize_parameters,
                        optimize_operators, optimize_fields)

                    top_results.append({
                        'rank': i + 1,
                        'score': trial.value,
                        'expression': trial_expression,
                        'params': trial.params,
                        'trial': trial
                    })

                # 保持向后兼容性：best_* 字段仍然指向第一个（最佳）结果
                best_result = top_results[0] if top_results else None

                return {
                    'best_score':
                    best_result['score'] if best_result else float('-inf'),
                    'best_expression':
                    best_result['expression'] if best_result else expression,
                    'best_params':
                    best_result['params'] if best_result else {},
                    'study':
                    study,
                    'n_trials':
                    len(study.trials),
                    'top_n_results':
                    top_results,  # 前N个结果
                    'top_n':
                    top_n
                }
            else:
                return {
                    'best_score': float('-inf'),
                    'best_expression': expression,
                    'best_params': {},
                    'study': study,
                    'n_trials': len(study.trials),
                    'top_n_results': [],
                    'top_n': top_n
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

    def _get_top_n_trials(self,
                          study,
                          top_n: int,
                          multi_objective: bool = False) -> List[Any]:
        """获取前N个最佳试验结果"""
        if multi_objective:
            # 多目标优化：返回帕累托前沿的前N个解
            pareto_front = study.best_trials
            return pareto_front[:top_n] if pareto_front else []
        else:
            # 单目标优化：按目标值排序，返回前N个
            trials = [t for t in study.trials if t.value is not None]
            trials.sort(key=lambda x: x.value, reverse=True)
            return trials[:top_n]

    def _reconstruct_expression_from_trial(self, original_expression: str,
                                           trial: Any,
                                           optimize_parameters: bool,
                                           optimize_operators: bool,
                                           optimize_fields: bool = False) -> str:
        """从试验结果重构表达式"""
        expression = original_expression

        # 重构参数
        if optimize_parameters:
            param_suggestions = {
                k: v
                for k, v in trial.params.items() if k.startswith('param_')
            }
            expression = self.param_optimizer.replace_parameters(
                expression, param_suggestions)

        # 重构算子
        if optimize_operators:
            op_replacements = {}
            for k, v in trial.params.items():
                if k.startswith('new_') and v is not None:
                    old_op = k.replace('new_', '')
                    op_replacements[old_op] = v
            expression = self.op_replacer.replace_operators(
                expression, op_replacements)

        # 重构特征
        if optimize_fields:
            field_replacements = {}
            for k, v in trial.params.items():
                if k.startswith('new_field_') and v is not None:
                    old_field = k.replace('new_field_', '')
                    field_replacements[old_field] = v
            expression = self.field_optimizer.replace_fields(
                expression, field_replacements)

        return expression

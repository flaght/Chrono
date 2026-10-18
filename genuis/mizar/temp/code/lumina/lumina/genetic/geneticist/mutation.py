import pdb, itertools, copy, itertools, optuna
from typing import List, Dict, Any
import pandas as pd
from ultron.utilities.logger import kd_logger
from ultron.factor.genetic.geneticist.operators import Operators
from ultron.factor.genetic.geneticist.operators import *
from lumina.genetic.process import *
from lumina.genetic.geneticist.operators import Operators as LOperators
from lumina.genetic.geneticist.express import ExpressionNode, ExpressionParser
from lumina.genetic.geneticist.blueprint import Blueprint
'''
class Converter(object):
    """
    一个能够将因子公式字符串，动态地、不依赖预生成实例地，
    转换为您的Program框架所需的“前缀表示法”列表的转换器。
    """

    def __init__(self, operator_template_map: Dict[str, Any]):
        """
        初始化转换器。

        :param operator_template_map: dict, 一个从函数名字符串到其“模板”Function对象的映射。
        """
        self.template_map = operator_template_map

    def _find_and_instantiate_operator(
            self, func_name: str, param_nodes: List[ExpressionNode]) -> Any:
        """
        根据函数名和参数节点，查找模板并动态创建一个带正确参数的Function实例。
        """
        template = self.template_map.get(func_name)
        if template is None:
            raise ValueError(f"在模板库中找不到函数: '{func_name}'")

        # 这是一个关键的假设：我们根据您的Function结构，认为其核心参数
        # (如周期) 是第一个parameter类型的子节点，并将其值赋给新实例的default_value。
        # 如果您的函数有多个参数，这里的逻辑需要扩展。
        new_default_value = None
        if param_nodes:
            new_default_value = param_nodes[0].value

        # 使用模板克隆并创建带有正确参数的新实例
        factors_instance = copy.deepcopy(
            template)  #.clone_with_params(new_default_value=new_default_value)
        factors_instance.default_value = new_default_value
        return factors_instance

    def _tree_to_prefix_recursive(self, node: ExpressionNode) -> List[Any]:
        """
        【核心】递归地将AST树转换为您的Program期望的前缀表示法列表。
        """
        # --- 基本情况: 如果是叶子节点 (特征或参数)，直接返回值 ---
        if not node.children:
            return [node.value]

        # --- 递归情况: 如果是函数节点 ---
        prefix_list = []

        # 1. 首先，添加当前的操作符 (动态实例化的Function对象)
        if node.node_type == 'function':
            param_nodes = [
                child for child in node.children
                if child.node_type == 'parameter'
            ]
            function_instance = self._find_and_instantiate_operator(
                node.value, param_nodes)
            prefix_list.append(function_instance)
        else:
            # 理论上不会发生，因为有子节点的都是函数节点
            raise TypeError(f"节点 {node.value} 有子节点但不是函数类型。")

        # 2. 然后，严格按照原始顺序，递归地添加所有子节点的列表
        for child in node.children:
            prefix_list.extend(self._tree_to_prefix_recursive(child))

        return prefix_list

    def formula_to_program_list(self, formula_string: str) -> List[Any]:
        """
        主接口函数：将完整的因子公式字符串转换为可被您Program框架执行的列表。
        """
        if not isinstance(formula_string, str) or not formula_string:
            raise ValueError("输入的公式必须是一个非空字符串。")

        # 步骤A：先用通用解析器将字符串解析成AST树
        parser = ExpressionParser(formula_string)
        ast_tree = parser.parse()

        # 步骤B：再用我们的递归转换器将AST树转换为Program所需的“前缀表示法”列表
        program_list = self._tree_to_prefix_recursive(ast_tree)

        return program_list
'''

MAX_INT = np.iinfo(np.int32).max
MIN_INT = np.iinfo(np.int32).min


class Converter:
    """
    一个能够将因子公式字符串，精确地转换为您的Program框架所需的、
    “参数被吸收的前缀表示法”列表的转换器。
    """

    def __init__(self, operator_template_map: Dict[str, Any]):
        self.template_map = operator_template_map

    def _find_and_instantiate_operator(
            self, func_name: str, param_nodes: List['ExpressionNode']) -> Any:
        template = self.template_map.get(func_name)
        if template is None:
            raise ValueError(f"在模板库中找不到函数模板: '{func_name}'")

        # 核心逻辑：将解析出的第一个参数值，赋给新实例的default_value
        new_default_value = template.default_value
        if param_nodes:
            new_default_value = param_nodes[0].value

        # 使用模板克隆并创建带有正确参数的新实例
        factors_instance = copy.deepcopy(
            template)  #.clone_with_params(new_default_value=new_default_value)
        factors_instance.default_value = new_default_value
        return factors_instance

    def _tree_to_prefix_recursive(self, node: 'ExpressionNode') -> List[Any]:
        """
        【最终正确版 V4】递归地将AST树转换为参数被吸收的前缀列表。
        """
        prefix_list = []

        # --- 1. 处理当前节点 ---
        if node.node_type == 'function':
            # a. 找到所有的参数子节点
            param_nodes = [
                child for child in node.children
                if child.node_type == 'parameter'
            ]
            # b. 使用参数来动态实例化一个带有正确 default_value 的 Function 对象
            function_instance = self._find_and_instantiate_operator(
                node.value, param_nodes)
            # c. 将这个完整的操作符，首先放入列表
            prefix_list.append(function_instance)

        elif node.node_type == 'feature':
            # 如果是特征，它是一个操作数，直接放入列表
            prefix_list.append(node.value)

        elif node.node_type == 'parameter':
            # 参数节点完全不进入最终列表，因为它们的信息已经被函数节点吸收了
            return []

        # --- 2. 递归处理所有非参数的子节点 ---
        non_param_children = [
            child for child in node.children if child.node_type != 'parameter'
        ]
        for child in non_param_children:
            prefix_list.extend(self._tree_to_prefix_recursive(child))

        return prefix_list

    def convert(self, formula_string: str) -> List[Any]:
        """
        主接口函数。
        """
        if not isinstance(formula_string, str) or not formula_string:
            raise ValueError("输入的公式必须是一个非空字符串。")

        parser = ExpressionParser(formula_string)
        ast_tree = parser.parse()

        return self._tree_to_prefix_recursive(ast_tree)


class Generator(object):
    """
    一个灵活的策略“配方”生成器，支持两种模式。
    """

    def __init__(self, signal_functions, strategy_functions):
        self.signal_functions = signal_functions
        self.strategy_functions = strategy_functions

        ## 转化器
        self.converter = Converter({
            op.name: op
            for op in Operators()._mutation_sets + Operators()._crossover_sets
        })

    def _generate_parameter_combinations(self, params_space):
        if not params_space: return [{}]
        return [
            dict(zip(params_space.keys(), combo))
            for combo in itertools.product(*params_space.values())
        ]

    # --- 模式一：参数调优的核心逻辑 ---
    def _generate_formula_param_variants(self, factor_tree,
                                         params_space) -> List[str]:

        def recursive_param_combination(node):
            if node.node_type != 'function':
                yield node
                return

            func_name = node.value
            param_indices = [
                i for i, child in enumerate(node.children)
                if child.node_type == 'parameter'
            ]
            other_indices = [
                i for i, child in enumerate(node.children)
                if child.node_type != 'parameter'
            ]

            other_variants_gen = [
                recursive_param_combination(node.children[i])
                for i in other_indices
            ]

            func_space = params_space.get(func_name, {})
            param_names = [f'param_{i}' for i in range(len(param_indices))]
            param_values = [
                func_space.get(p_name, [node.children[p_idx].value])
                for p_idx, p_name in zip(param_indices, param_names)
            ]
            direct_param_combos = list(itertools.product(*param_values))

            for other_combo in itertools.product(*other_variants_gen):
                for param_combo in direct_param_combos:
                    new_node = copy.deepcopy(node)
                    new_node.children = [None] * len(node.children)
                    for i, p_idx in enumerate(param_indices):
                        new_node.children[p_idx] = ExpressionNode(
                            param_combo[i], 'parameter')
                    for i, o_idx in enumerate(other_indices):
                        new_node.children[o_idx] = other_combo[i]
                    yield new_node

        return [str(tree) for tree in recursive_param_combination(factor_tree)]

    def tune_parameters(self, base_info, factor_params_space):
        kd_logger.info("--- 模式一：开始对固定结构进行参数调优 ---")
        parser = ExpressionParser(base_info['formual'])
        factor_tree = parser.parse()
        components = factor_tree.extract_components()
        formula_variants = self._generate_formula_param_variants(
            factor_tree, factor_params_space)
        kd_logger.info(f"生成了 {len(formula_variants)} 种因子公式参数变体。")

        all_variants = []
        for formual, signal, strategy in itertools.product(
                formula_variants, self.signal_functions,
                self.strategy_functions):
            all_variants.append({
                'formual': formual,
                'program': self.converter.convert(formual),
                'features': components['features'],
                'signal_method': signal,
                'signal_name': signal.name,
                'signal_params': signal.params,
                'strategy_method': strategy,
                'strategy_name': strategy.name,
                'strategy_params': strategy.params,
            })

        kd_logger.info(f"总共生成 {len(all_variants)} 个策略配方。")
        return all_variants

    # --- 模式二：结构探索的核心逻辑 ---
    def _generate_factor_formulas_by_depth(self, factor_ops,
                                           base_features: List[str],
                                           max_depth: int) -> List[str]:
        unary_ops = [op for op in factor_ops if op.arity == 1]
        binary_ops = [op for op in factor_ops if op.arity == 2]
        features = [f"'{f}'" for f in base_features]
        formulas = {0: features}

        for d in range(1, max_depth + 1):
            formulas[d] = []
            if d - 1 in formulas:
                for op in unary_ops:
                    for sub_f in formulas[d - 1]:
                        formulas[d].append(
                            f"{op.name}({op.default_value}, {sub_f})")
            for op in binary_ops:
                for i in range(d):
                    j = d - 1 - i
                    if i in formulas and j in formulas:
                        for f1 in formulas[i]:
                            for f2 in formulas[j]:
                                formulas[d].append(f"{op.name}({f1}, {f2})")

        all_f = []
        for d in range(1, max_depth + 1):
            all_f.extend(formulas.get(d, []))
        return list(set(all_f))

    def explore_structures(self, base_info, period_params, factor_max_depth):
        parser = ExpressionParser(base_info['formual'])
        factor_tree = parser.parse()
        components = factor_tree.extract_components()
        base_features = components['features']
        factor_operators = LOperators().create_operators(
            operators_sets=components['functions'],
            period_params=period_params)

        kd_logger.info(f"--- 模式二：开始探索深度为1到{factor_max_depth}的策略结构 ---")
        formula_variants = self._generate_factor_formulas_by_depth(
            factor_operators, base_features, factor_max_depth)
        kd_logger.info(f"共生成 {len(formula_variants)} 个独特的因子公式。")

        all_variants = []
        for formual, signal, strategy in itertools.product(
                formula_variants, self.signal_functions,
                self.strategy_functions):
            all_variants.append({
                'formual': formual,
                'features': components['features'],
                'program': self.converter.convert(formual),
                'signal_method': signal.name,
                'signal_params': signal.params,
                'strategy_method': strategy.name,
                'strategy_params': strategy.params
            })

        all_variants = [
            Blueprint(fitness=0,
                      coverage_rate=0.8,
                      program=variant['program'],
                      signal_method=variant['signal_method'],
                      strategy_method=variant['strategy_method'])
            for variant in all_variants
        ]
        kd_logger.info(f"总共生成 {len(all_variants)} 个策略配方。")
        return all_variants


def create_fitness(column, configure, total_data):
    column.raw_fitness(total_data=total_data,
                       factor_sets=column._factor_sets,
                       custom_params=configure['custom_params'],
                       backup_cycle=1,
                       default_value=MIN_INT)
    return column


### 批量计算绩效
@add_process_env_sig
def run_fitness(target_column, configure, total_data):
    position_data = run_process(target_column=target_column,
                                callback=create_fitness,
                                configure=configure,
                                total_data=total_data)
    return position_data


class Actuator(object):

    def __init__(self, callback_fitness, k_split=1):
        self.k_split = k_split
        self.callback_fitness = callback_fitness

    def calculate(self, strategies_infos, total_data, factor_columns,
                  configure):
        all_variants = [
            Blueprint(fitness=self.callback_fitness,
                      method='full',
                      coverage_rate=configure['coverage_rate'],
                      factor_sets=factor_columns,
                      program=variant['program'],
                      signal_method=variant['signal_method'],
                      strategy_method=variant['strategy_method'])
            for variant in strategies_infos
        ]
        return self._calculate(strategies_infos=all_variants,
                               total_data=total_data,
                               configure=configure)

    def _calculate(self, strategies_infos, total_data, configure):
        process_list = split_k(self.k_split, strategies_infos)
        population = create_parellel(process_list=process_list,
                                     callback=run_fitness,
                                     configure=configure,
                                     total_data=total_data)
        population = list(itertools.chain.from_iterable(population))
        return population


class Optimizer(object):

    def __init__(self,
                 actuator,
                 total_data,
                 configure,
                 search_rules,
                 signals_sets,
                 strategies_sets,
                 factor_columns,
                 callback_fitness,
                 k_split=1):
        self.actuator = actuator
        self.total_data = total_data
        self.configure = configure
        self.search_rules = search_rules  # 包含了如何生成搜索空间的规则

        self.operators_sets = Operators()._mutation_sets + Operators(
        )._crossover_sets
        self.signals_sets = signals_sets
        self.strategies_sets = strategies_sets
        self.factor_columns = factor_columns
        self.callback_fitness = callback_fitness
        self.k_split = k_split
        self.converter = Converter({op.name: op for op in self.operators_sets})

    def _create_optuna_space1(self, trial, original_params):
        params = {}
        for p_name, p_val in original_params.items():
            rule_key = p_name
            if not rule_key in self.search_rules:
                parts = p_name.split('_')
                rule_key = parts[1] if len(parts) > 1 else 'default'
            rule = self.search_rules.get(rule_key,
                                         self.search_rules['default'])
            low, high = p_val * rule['range_pct'][0], p_val * rule[
                'range_pct'][1]
            step = rule.get('step')
            if isinstance(p_val, int):
                params[p_name] = trial.suggest_int(
                    p_name, int(low), int(high), step=int(step) if step else 1)
            else:
                params[p_name] = trial.suggest_float(p_name,
                                                     low,
                                                     high,
                                                     step=step)
        return params

    def _create_optuna_space2(self,
                              trial: optuna.Trial,
                              center_params: dict,
                              phase: str = 'coarse'):
        """
        【核心升级 V2】根据阶段和层级化参数名，动态生成Optuna搜索空间。
        """
        params = {}
        for p_name, p_val in center_params.items():

            # 1. 尝试用全名精确匹配
            rule = self.search_rules.get(p_name)

            # 2. 如果全名匹配失败，则解析参数名
            if rule is None:
                parts = p_name.split('_')
                # 倒序遍历parts，寻找第一个在search_rules中存在的键
                # e.g., for 'fac_MIChimoku_RSI_p0', it will check 'RSI', then 'MIChimoku'
                # parts without 'fac', 'sig', 'str' and 'pX'
                potential_keys = [
                    part for part in parts
                    if part not in ['fac', 'sig', 'str']
                    and not part.startswith('p')
                ]

                for key in reversed(potential_keys):
                    if key in self.search_rules:
                        rule = self.search_rules[key]
                        break  # 找到最内层的匹配后就停止

            # 3. 如果所有部分都匹配失败，则使用默认规则
            if rule is None:
                rule = self.search_rules['default']

            # ===============================================

            # 根据阶段选择范围和步长 (这部分逻辑不变)
            range_pct_key = 'range_pct' if phase == 'coarse' else 'fine_range_pct'
            step_key = 'step' if phase == 'coarse' else 'fine_step'

            # 确保即使规则不完整，也有默认值
            range_pct = rule.get(range_pct_key,
                                 self.search_rules['default'][range_pct_key])
            step = rule.get(step_key)  # step可以是None

            low = p_val * range_pct[0]
            high = p_val * range_pct[1]

            if isinstance(p_val, int):
                # 确保类型正确和范围有效
                low_int = int(round(low))
                high_int = int(round(high))
                step_int = int(round(step)) if step is not None and isinstance(
                    step, (int, float)) else 1
                # 确保下限不小于一个合理值，例如1
                low = max(1, low_int)
                # 保证上限至少比下限大一个步长
                high_int = max(low_int, high_int)
                # 如果low和high相等，无法采样，我们将其扩展一个步长的范围
                if low_int >= high_int:
                    high_int = low_int + step_int
                params[p_name] = trial.suggest_int(p_name,
                                                   low=low_int,
                                                   high=high_int,
                                                   step=step_int)
            elif isinstance(p_val, float):  # float
                step_float = float(step) if step is not None and isinstance(
                    step, (int, float)) else None
                # 保证上限大于下限
                if low >= high:
                    high = low + (step_float if step_float else 1e-6)
                params[p_name] = trial.suggest_float(p_name,
                                                     low=low,
                                                     high=high,
                                                     step=step_float)
            else:
                # 对于其他未知类型，直接使用中心值，不进行优化
                kd_logger.warning(
                    f"Warning: Parameter '{p_name}' has an unsupported type ({type(p_val)}). Using its original value."
                )
                params[p_name] = p_val

        return params

    def _parse_formula_params(self, factor_tree):
        params = {}

        def recurse(node, context=""):
            if node.node_type == 'function':
                func_context = f"{context}{node.value}_"
                param_nodes = [
                    c for c in node.children if c.node_type == 'parameter'
                ]
                for i, p_node in enumerate(param_nodes):
                    params[f"factor_{func_context}p{i}"] = p_node.value
                for child in node.children:
                    if child.node_type == 'function':
                        recurse(child, func_context)

        recurse(factor_tree)
        return params

    def _rebuild_formula_from_params(self, factor_tree, params):
        tree_copy = copy.deepcopy(factor_tree)

        def recurse(node, context=""):
            if node.node_type == 'function':
                func_context = f"{context}{node.value}_"
                param_nodes = [
                    c for c in node.children if c.node_type == 'parameter'
                ]
                for i, p_node in enumerate(param_nodes):
                    param_name = f"factor_{func_context}p{i}"
                    if param_name in params: p_node.value = params[param_name]
                for child in node.children:
                    if child.node_type == 'function':
                        recurse(child, func_context)

        recurse(tree_copy)
        return str(tree_copy)

    def _generate_recipe_from_flat_params(self, flat_params, base_info,
                                          factor_tree):
        factor_params = {
            k: v
            for k, v in flat_params.items() if k.startswith('factor_')
        }
        signal_params = {
            k.replace('signal_', ''): v
            for k, v in flat_params.items() if k.startswith('signal_')
        }
        strategy_params = {
            k.replace('strategy_', ''): v
            for k, v in flat_params.items() if k.startswith('strategy_')
        }
        
        formula = self._rebuild_formula_from_params(factor_tree, factor_params)


        ## 查找模版
        signal_func_template = next(s for s in self.signals_sets
                           if s.name == base_info['signal_method'])
        strategy_func_template = next(s for s in self.strategies_sets if s.name == base_info['strategy_method'])
        
         # 创建带新参数的副本
        signal_func_final = copy.deepcopy(signal_func_template)
        signal_func_final.params.update(signal_params)
        
        strategy_func_final = copy.deepcopy(strategy_func_template)
        strategy_func_final.params.update(strategy_params)
        
        return {
            'formual': formula,
            'signal_method': signal_func_final,
            'strategy_method': strategy_func_final
        }

    def _create_blueprints_from_recipes(self, recipes):
        blueprints = []
        for recipe in recipes:
            try:
                program_list = self.converter.convert(recipe['formual'])
                blueprints.append(
                    Blueprint(fitness=self.callback_fitness,
                              method='full',
                              factor_sets=self.factor_columns,
                              coverage_rate=self.configure['coverage_rate'],
                              program=program_list,
                              signal_method=recipe['signal_method'],
                              strategy_method=recipe['strategy_method']))
            except Exception as e:
                kd_logger.warning(f"创建Blueprint失败: {e}")
        return blueprints

    def _evaluate_recipes(self, recipes):
        if not recipes: return pd.DataFrame()
        blueprints = self._create_blueprints_from_recipes(recipes)
        if not blueprints: return pd.DataFrame()
        evaluated_bps = self.actuator._calculate(strategies_infos=blueprints,
                                                 total_data=self.total_data,
                                                 configure=self.configure)
        return pd.DataFrame([p.output() for p in evaluated_bps])

    def _evaluate_recipe(self, recipe):
        if not recipe: return pd.DataFrame()
        blueprints = self._create_blueprints_from_recipes([recipe])
        if not blueprints: return pd.DataFrame()
        res = []
        for blueprint in blueprints:
            blueprint.raw_fitness(
                total_data=self.total_data,
                factor_sets=blueprint._factor_sets,
                custom_params=self.configure['custom_params'],
                backup_cycle=1,
                default_value=MIN_INT)
            res.append(blueprint.output())

        return pd.DataFrame(res)

  
    def optimize1(self,
                  strategy_info: dict,
                  n_trials: int = 100,
                  top_n_results: int = 10):
        kd_logger.info(
            f"\n{'='*20} Optimizing Strategy: {strategy_info.get('name', 'Unnamed')} {'='*20}"
        )

        parser = ExpressionParser(strategy_info['formual'])
        factor_tree = parser.parse()
        original_params = {}
        original_params.update(self._parse_formula_params(factor_tree))
        original_params.update({
            f'signal_{k}': v
            for k, v in strategy_info['signal_params'].items()
        })
        original_params.update({
            f'strategy_{k}': v
            for k, v in strategy_info['strategy_params'].items()
            if k != 'max_volume'
        })

        def objective(trial: optuna.Trial):
            params_to_test = self._create_optuna_space1(trial, original_params)
            recipe = self._generate_recipe_from_flat_params(
                params_to_test, strategy_info, factor_tree)
            result_df = self._evaluate_recipes([recipe])
            return result_df['final_fitness'].iloc[
                0] if not result_df.empty else -np.inf

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        best_params_coarse = study.best_params
        kd_logger.info("\n--- 阶段二：精细化穷举 ---")
        fine_grained_space = {}
        for p_name, p_val in best_params_coarse.items():
            rule_key = p_name
            if not rule_key in self.search_rules:
                parts = p_name.split('_')
                rule_key = parts[1] if len(parts) > 1 else 'default'
            rule = self.search_rules.get(rule_key,
                                         self.search_rules['default'])
            fine_range = np.array(p_val) * np.array(
                rule.get('fine_range_pct', [0.9, 1.1]))
            fine_step = rule.get('fine_step',
                                 0.1 if isinstance(p_val, float) else 1)
            if isinstance(p_val, int):
                fine_grained_space[p_name] = range(int(fine_range[0]),
                                                   int(fine_range[1]) + 1,
                                                   int(fine_step))
            else:
                fine_grained_space[p_name] = np.arange(
                    fine_range[0], fine_range[1] + fine_step, fine_step)

        param_grid = list(itertools.product(*fine_grained_space.values()))
        flat_params_list = [
            dict(zip(fine_grained_space.keys(), combo)) for combo in param_grid
        ]
        recipes_to_test = [
            self._generate_recipe_from_flat_params(p, strategy_info,
                                                   factor_tree)
            for p in flat_params_list
        ]

        kd_logger.info(f"将对 {len(recipes_to_test)} 个精细化策略进行最终批量回测...")
        results_df = self._evaluate_recipes(recipes_to_test)

        if results_df.empty: return pd.DataFrame()

        # 将参数与结果合并
        results_with_params = pd.concat([
            pd.DataFrame(flat_params_list),
            results_df.reset_index(drop=True)
        ],
                                        axis=1)
        top_results = results_with_params.sort_values(
            'final_fitness', ascending=False).head(top_n_results)

        kd_logger.info(f"\n--- 参数优化完成，返回Top {top_n_results} 结果 ---")
        return top_results
    

    def optimize2(
            self,
            strategy_info: dict,
            coarse_n_trials: int = 100,  # 粗调阶段的试验次数
            fine_n_trials: int = 200,  # 精调阶段的试验次数
            top_n_results: int = 10):

        kd_logger.info(
            f"\n{'='*20} Optimizing Strategy: {strategy_info.get('name', 'Unnamed')} {'='*20}"
        )

        # --- 准备工作：解析种子策略，得到AST和原始参数 ---
        parser = ExpressionParser(strategy_info['formual'])
        factor_tree = parser.parse()
        original_params = {}
        original_params.update(self._parse_formula_params(factor_tree))
        original_params.update({
            f'signal_{k}': v
            for k, v in strategy_info['signal_params'].items()
        })
        # 排除固定的max_volume
        original_params.update({
            f'strategy_{k}': v
            for k, v in strategy_info['strategy_params'].items()
            if k != 'max_volume'
        })

        def objective(trial: optuna.Trial, phase: str, center_params: dict):
            params_to_test = self._create_optuna_space2(
                trial, center_params, phase)
            recipe = self._generate_recipe_from_flat_params(
                params_to_test, strategy_info, factor_tree)
            # 直接在这里将配方转换为Blueprint并评估
            blueprints = self._create_blueprints_from_recipes([recipe])
            blueprint = blueprints[0]
            if not blueprint: return -np.inf
            blueprint.raw_fitness(total_data=self.total_data,
                       factor_sets=blueprint._factor_sets,
                       custom_params=self.configure['custom_params'],
                       backup_cycle=1,
                       default_value=MIN_INT)
            return blueprint._final_fitness if blueprints else -np.inf
            '''
            evaluated = self.actuator.calculate(blueprints, self.total_data,
                                                self.configure)
            return evaluated[0].final_fitness if evaluated else -np.inf
            '''

        # ==============================================================================
        # 阶段一：Optuna 粗粒度全局探索
        # ==============================================================================
        kd_logger.info("\n--- 阶段一：Optuna 粗粒度全局探索 ---")
        '''
        def objective_coarse(trial: optuna.Trial):
            # 动态生成“大范围、大步长”的搜索空间
            params_to_test = self._create_optuna_space2(trial,
                                                        original_params,
                                                        phase='coarse')
            recipe = self._generate_recipe_from_flat_params(
                params_to_test, strategy_info, factor_tree)
            result_df = self._evaluate_recipe(recipe)
            return result_df['final_fitness'].iloc[
                0] if not result_df.empty else -np.inf
        '''

        study_coarse = optuna.create_study(direction='maximize')
        #study_coarse.optimize(objective_coarse,
        #                      n_trials=coarse_n_trials,
        #                      n_jobs=self.k_split)
        study_coarse.optimize(
            lambda t: objective(t, 'coarse', original_params),
            n_trials=coarse_n_trials,
            n_jobs=self.k_split)

        best_params_coarse = study_coarse.best_params
        kd_logger.info(f"阶段一完成。找到的最佳粗粒度参数: {best_params_coarse}")

        # ==============================================================================
        # 阶段二：Optuna 细粒度局部精炼
        # ==============================================================================

        kd_logger.info("\n--- 阶段二：围绕最优区域进行Optuna精细化搜索 ---")
        '''
        def objective_fine(trial: optuna.Trial):
            # 动态生成“小范围、小步长”的搜索空间
            params_to_test = self._create_optuna_space2(
                trial, best_params_coarse, phase='fine')  # 注意：中心点是第一阶段的最优参数
            recipe = self._generate_recipe_from_flat_params(
                params_to_test, strategy_info, factor_tree)
            result_df = self._evaluate_recipes([recipe])
            return result_df['final_fitness'].iloc[
                0] if not result_df.empty else -np.inf
        '''

        study_fine = optuna.create_study(direction='maximize')
        #study_fine.optimize(objective_fine,
        #                    n_trials=fine_n_trials,
        #                    n_jobs=self.k_split)
        study_fine.optimize(
            lambda t: objective(t, 'fine', best_params_coarse),
            n_trials=fine_n_trials,
            n_jobs=self.k_split)
        # --- 邻里审查 & 最终返回Top-N结果 ---
        kd_logger.info("\n--- 优化完成，整理并返回Top-N结果 ---")

        # 从第二次（精细化）的搜索结果中提取所有试验
        completed_trials = [
            t for t in study_fine.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if not completed_trials:
            kd_logger.error("精细化调优未能产生任何有效策略。")
            return pd.DataFrame()

        # 按性能对所有成功的试验进行排序
        completed_trials.sort(key=lambda t: t.value, reverse=True)

        top_blueprints = []
        # 只取前N个
        for trial in completed_trials[:top_n_results]:
            best_params = trial.params
            # 使用找到的最佳参数，重新生成最终的“配方”
            final_recipe = self._generate_recipe_from_flat_params(
                best_params, strategy_info, factor_tree)
            # 将这个最佳配方，实例化为Blueprint对象
            # _create_blueprints_from_recipes返回列表，我们取第一个
            blueprint_instance = self._create_blueprints_from_recipes(
                [final_recipe])[0]
            # 将Optuna找到的fitness分数，回填到Blueprint对象中
            blueprint_instance._final_fitness = trial.value
            blueprint_instance._raw_fitness = trial.value
            top_blueprints.append(blueprint_instance)

        kd_logger.info(f"成功构建了 {len(top_blueprints)} 个最优的Blueprint对象。")
        return top_blueprints
        '''
        # 将所有试验结果转换为DataFrame
        results_list = []
        for trial in completed_trials:
            result_row = trial.params.copy()
            result_row['fitness'] = trial.value
            results_list.append(result_row)

        results_df = pd.DataFrame(results_list)

        top_results = results_df.sort_values(
            'fitness', ascending=False).head(top_n_results)

        kd_logger.info(f"返回Top {top_n_results} 个最佳参数组合及其绩效。")
        return top_results
        '''

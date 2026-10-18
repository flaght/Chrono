# -*- coding: utf-8 -*-
import pdb
import numpy as np
import time, datetime, hashlib, copy
from ultron.utilities.logger import kd_logger
from ultron.factor.genetic.geneticist.operators import crossover_sets, mutation_sets, calc_factor, Function, FunctionType
from lumina.genetic.util import create_id

import warnings

warnings.filterwarnings("ignore")

ABS_FLOAT = 0.000001


class Program(object):

    def __init__(self,
                 init_depth,
                 method,
                 random_state,
                 factor_sets,
                 p_point_replace,
                 function_set,
                 operators_set,
                 gen,
                 fitness,
                 coverage_rate=0.8,
                 n_features=0,
                 program=None,
                 parents=None,
                 operator_probs=None,
                 factor_probs=None):
        self._init_depth = init_depth
        self._init_method = method
        self._program = program
        self._factor_sets = factor_sets
        self._p_point_replace = p_point_replace
        self._function_set = function_set
        self._operators_set = operators_set
        self._n_features = n_features
        self._fitness = fitness
        self._coverage_rate = coverage_rate
        self._gen = gen
        self._raw_fitness = None  # fitness得分
        self._final_fitness = None  # 最终得分·
        self._max_corr = 0  # 最大相关性
        self._alpha = 0  # 惩罚系数alpha的类
        self._penalty = 0  # 惩罚系数
        self._is_valid = True
        self._parents = parents
        self._retain_data = None
        self._create_time = datetime.datetime.now()
        self._temp_id = 'ultron_' + str(
            int(time.time() * 1000000 + datetime.datetime.now().microsecond))
        self._name = self._temp_id

        self._operator_probs = operator_probs
        self._factor_probs = factor_probs

        self._factor_data = None
        if self._program is None:
            self._program = self.build_program(random_state)
        self.variation()
        self.create_identification()

    def parent_idx(self):
        return 0 if self._parents is None else self._parents['parent_idx']

    def penalty(self, penalty, max_corr, alpha):
        self._final_fitness -= penalty
        self._max_corr = max_corr
        self._penalty = penalty
        self._alpha = alpha

    def log(self):
        parents = {'method': 'gen'} if self._parents is None else self._parents
        formual = self.transform()
        identification = self._identification
        name = self._name
        kd_logger.info(
            'name:%s,method:%s,gen:%d,formual:%s,fitness:%f,identification:%s'
            % (name, str(parents['method']), self._gen, formual,
               self._final_fitness, identification))

    def expression(self):
        res = []
        for node in self._program:
            if self._is_function_node(node):
                res.append(node.name)
            else:
                res.append(node)
        return res

    def output(self):
        parents = {'method': 'Gen'} if self._parents is None else self._parents
        return {
            'name': self._name,
            'method': parents['method'],
            'gen': self._gen,
            'features': self._identification,
            'formual': self.transform(),
            'final_fitness': self._final_fitness,
            'raw_fitness': self._raw_fitness,
            'max_corr': self._max_corr,
            'penalty': self._penalty,
            'alpha': self._alpha,
            'update_time': self._create_time
        }

    # 交叉变异时会出生成无效子代,设置无效标识
    # 如 ['CurrentAssetsTRate', 'CurrentAssetsTRate', 'rskew_std']
    def variation(self):
        check_result = [p for p in self._program if self._is_function_node(p)]
        if len(check_result) == 0 and len(self._program) > 1:
            self._program = self._program[:1]
            self._is_valid = False
            return
        
        # 验证程序结构是否能生成有效的表达式
        if not self._validate_structure(self._program):
            self._is_valid = False
            kd_logger.debug(f"程序结构验证失败: {self._program[:5]}...")
    
    def _validate_structure(self, program):
        """验证程序结构是否能形成有效表达式
        
        程序使用前缀表达式存储：[Function, arg1, arg2, ...]
        这个方法模拟 transform() 的处理过程来验证结构
        
        Args:
            program: 程序列表
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not program:
            return False
        
        # 单个因子是有效的
        if len(program) == 1:
            return not self._is_function_node(program[0])
        
        # 多元素程序必须包含算子
        has_function = any(self._is_function_node(node) for node in program)
        if not has_function:
            return False
        
        # 模拟 transform() 的栈处理过程
        try:
            apply_stack = []
            node_index = 0
            
            for node_index, node in enumerate(program):
                if self._is_function_node(node):
                    apply_stack.append([node])
                else:
                    # 因子需要追加到最后一个栈
                    if not apply_stack:
                        return False  # 没有栈可追加，无效
                    apply_stack[-1].append(node)
                
                # 检查是否有栈满足条件（参数已足够）
                while apply_stack and len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                    # 弹出完整的栈
                    if len(apply_stack) != 1:
                        apply_stack.pop()
                        if not apply_stack:
                            return False
                        # 将结果追加到上一层（这里用None代表结果）
                        apply_stack[-1].append(None)
                    else:
                        # 只剩一个栈且已完成
                        # 检查是否还有剩余节点（孤立节点）
                        if node_index < len(program) - 1:
                            return False  # 还有剩余节点，无效
                        return True
            
            # 遍历结束后还有未完成的栈，说明参数不足
            return False
        except Exception:
            return False
    
    def _is_function_node(self, node):
        """判断节点是否为函数/算子节点
        
        兼容 Function 类和具有 arity 属性的类（用于测试）
        """
        return (isinstance(node, Function) or 
                (hasattr(node, 'arity') and hasattr(node, 'function')))


    def create_identification(self):
        m = hashlib.md5()
        try:
            token = self.transform()
        except Exception as e:
            #ID为key
            token = self._temp_id
        if token is None:
            token = self._temp_id
        m.update(bytes(token, encoding='UTF-8'))
        self._identification = m.hexdigest()
        ## 重新构建name
        name = "ultron_{0}".format(create_id(original=self._identification, digit=16))
        if self._name == self._temp_id:
            self._name = name

    def create_formual(self, apply_formual):
        function = apply_formual[0]
        formula = function.function.__name__
        if function.ftype == FunctionType.cross_section:
            formula += '('
        else:
            formula += ('(' + str(function.default_value) + ',')
        for i in range(0, function.arity):
            if i != 0:
                formula += ','
            if apply_formual[i + 1] in self._factor_sets:
                formula += '\'' + apply_formual[i + 1] + '\''
            else:
                formula += apply_formual[i + 1]
        formula += ')'
        return formula

    def transform(self):
        if len(self._program) < 2:
            result = 'CURRENT(\'' + self._program[0] + '\')'
            return result
        apply_stack = []
        for node in self._program:
            if self._is_function_node(node):
                apply_stack.append([node])
            else:
                try:
                    apply_stack[-1].append(node)
                except Exception as e:
                    return None
            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                result = self.create_formual(apply_stack[-1])
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(result)
                else:
                    return result

    def export_graphviz(self):
        fade_nodes = None
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self._program):
            fill = '#cecece'
            if self._is_function_node(node):
                if i not in fade_nodes:
                    fill = '#2a5caa'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n' %
                           (i, node.function.__name__, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if node in self._factor_sets:
                    feature_name = node
                else:
                    feature_name = 'X%s' % node
                output += ('%d [label="%s", fillcolor="%s"] ;\n' %
                           (i, feature_name, fill))

                if i == 0:
                    output += '}'
                    return output
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            output += '}'
                            return output
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

    def build_program(self, random_state):
        #在范围内选取树形深度
        if self._init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self._init_method
        if isinstance(self._init_depth, int):
            max_depth = self._init_depth
        else:
            max_depth = random_state.randint(*self._init_depth)

        if self._operator_probs is None:
            function = self._operators_set[random_state.randint(
                len(self._operators_set))]
        else:
            function = random_state.choice(self._operators_set,
                                           p=self._operator_probs)
        program = [function]
        terminal_stack = [function.arity]
        while terminal_stack:
            depth = len(terminal_stack)
            choice = self._n_features + len(self._operators_set)
            choice = np.random.randint(0, choice)
            if depth < max_depth and (method == 'full'
                                      or choice <= len(self._operators_set)):
                if self._operator_probs is None:
                    function = self._operators_set[np.random.randint(
                        0,
                        len(self._operators_set) - 1)]
                else:
                    function = random_state.choice(self._operators_set,
                                                   p=self._operator_probs)
                program.append(function)
                terminal_stack.append(function.arity)
            else:
                if self._factor_probs is None:
                    factor = self._factor_sets[np.random.randint(
                        0,
                        len(self._factor_sets) - 1)]
                else:
                    factor = random_state.choice(self._factor_sets,
                                                 p=self._factor_probs)
                program.append(factor)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1
        return program

    def get_subtree(self, random_state, program=None):
        if program is None:
            program = self._program
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array(
            [0.9 if node in self._operators_set else 0.1 for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())
        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if node in self._operators_set:
                stack += node.arity
            end += 1
        return start, end

    ##复制
    def reproduce(self):
        return copy.deepcopy(self._program)

    ##交叉
    def crossover(self, donor, random_state, max_attempts=5):
        """交叉操作，增加验证和重试机制"""
        for attempt in range(max_attempts):
            try:
                start, end = self.get_subtree(random_state)
                end -= 1
                removed = range(start, end)
                donor_start, donor_end = self.get_subtree(random_state, donor)
                donor_removed = list(
                    set(range(len(donor))) - set(range(donor_start, donor_end)))
                program = (self._program[:start] + donor[donor_start:donor_end] +
                          self._program[end:])
                
                # 验证生成的程序
                if self._validate_structure(program):
                    return program, removed, donor_removed
                else:
                    if attempt < max_attempts - 1:
                        kd_logger.debug(f"交叉生成无效程序，重试 {attempt + 1}/{max_attempts}")
                        continue
            except Exception as e:
                if attempt < max_attempts - 1:
                    kd_logger.debug(f"交叉出错: {e}，重试 {attempt + 1}/{max_attempts}")
                    continue
        
        # 失败后返回原程序（不做交叉）
        kd_logger.debug("交叉操作多次失败，返回原程序")
        return self._program, [], []

    ##树变异
    def subtree_mutation(self, random_state, max_attempts=5):
        """子树变异，增加验证和重试机制"""
        for attempt in range(max_attempts):
            try:
                chicken = self.build_program(random_state)
                program, removed, donor_removed = self.crossover(chicken, random_state)
                
                # 验证生成的程序
                if self._validate_structure(program):
                    return program, removed, donor_removed
                else:
                    if attempt < max_attempts - 1:
                        kd_logger.debug(f"子树变异生成无效程序，重试 {attempt + 1}/{max_attempts}")
                        continue
            except Exception as e:
                if attempt < max_attempts - 1:
                    kd_logger.debug(f"子树变异出错: {e}，重试 {attempt + 1}/{max_attempts}")
                    continue
        
        # 失败后返回原程序（不做变异）
        kd_logger.debug("子树变异多次失败，返回原程序")
        return self._program, [], []

    ##突变异
    def hoist_mutation(self, random_state, max_attempts=5):
        """提升变异，增加验证和重试机制"""
        for attempt in range(max_attempts):
            try:
                start, end = self.get_subtree(random_state)
                subtree = self._program[start:end]
                sub_start, sub_end = self.get_subtree(random_state, subtree)
                hoist = subtree[sub_start:sub_end]
                removed = list(
                    set(range(start, end)) -
                    set(range(start + sub_start, start + sub_end)))
                program = self._program[:start] + hoist + self._program[end:]
                
                # 验证生成的程序
                if self._validate_structure(program):
                    return program, removed
                else:
                    if attempt < max_attempts - 1:
                        kd_logger.debug(f"提升变异生成无效程序，重试 {attempt + 1}/{max_attempts}")
                        continue
            except Exception as e:
                if attempt < max_attempts - 1:
                    kd_logger.debug(f"提升变异出错: {e}，重试 {attempt + 1}/{max_attempts}")
                    continue
        
        # 失败后返回原程序（不做变异）
        kd_logger.debug("提升变异多次失败，返回原程序")
        return self._program, []
    
    ##点变异
    def point_mutation(self, random_state, max_attempts=5):
        """点变异，增加验证和重试机制"""
        for attempt in range(max_attempts):
            try:
                program = copy.deepcopy(self._program)
                mutate = np.where(
                    random_state.uniform(size=len(program)) < self._p_point_replace)[0]

                for node in mutate:
                    if program[node] in self._operators_set:
                        activy = program[node].arity
                        #找到参数个数替换
                        if activy == 1:
                            replace_node = mutation_sets[random_state.randint(
                                0,
                                len(mutation_sets))]
                        else:
                            replace_node = crossover_sets[random_state.randint(
                                0,
                                len(crossover_sets))]
                        program[node] = replace_node
                    else:
                        if self._factor_probs is None:
                            factor = self._factor_sets[random_state.randint(
                                0,
                                len(self._factor_sets))]
                        else:
                            factor = random_state.choice(self._factor_sets,
                                                         p=self._factor_probs)

                        program[node] = factor
                
                # 验证生成的程序
                if self._validate_structure(program):
                    return program, list(mutate)
                else:
                    if attempt < max_attempts - 1:
                        kd_logger.debug(f"点变异生成无效程序，重试 {attempt + 1}/{max_attempts}")
                        continue
            except Exception as e:
                if attempt < max_attempts - 1:
                    kd_logger.debug(f"点变异出错: {e}，重试 {attempt + 1}/{max_attempts}")
                    continue
        
        # 失败后返回原程序（不做变异）
        kd_logger.debug("点变异多次失败，返回原程序")
        return self._program, []

    def raw_fitness(self,
                    total_data,
                    factor_sets,
                    default_value,
                    backup_cycle,
                    custom_params,
                    indexs=['trade_date'],
                    key='code'):
        #计算因子值
        if not self._is_valid:
            self._raw_fitness = default_value
            return
        try:
            expression = self.transform()
            if expression is None:
                self._raw_fitness = default_value
                self._is_valid = False
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ## 外部指定数据格式，set_index('trade_date') 在数据庞大时会出现性能问题
                    factor_data = calc_factor(expression, total_data, indexs,
                                              key)
                #切割掉备份周期
                factor_data = factor_data.replace([np.inf, -np.inf], np.nan)
                #处理因子暴露度为0
                factor_data['transformed'] = np.where(
                    np.abs(factor_data.transformed.values) > 0.000001,
                    factor_data.transformed.values, np.nan)
                factor_data = factor_data.loc[factor_data.index.unique()
                                              [backup_cycle:]]
                self._factor_data = factor_data  ## 保存因子值
                ##检测覆盖率
                coverage_rate = 1 - factor_data['transformed'].isna().sum(
                ) / len(factor_data['transformed'])
                self._raw_fitness = default_value
                if coverage_rate > self._coverage_rate:
                    #cycle_total_data = total_data.copy().set_index(
                    #    'trade_date')
                    cycle_total_data = total_data.copy()
                    cycle_total_data = cycle_total_data.loc[
                        cycle_total_data.index.unique()[backup_cycle:]]
                    new_custom_params = copy.deepcopy(custom_params)
                    new_custom_params['name'] = self._name
                    results = self._fitness(factor_data,
                                            cycle_total_data.reset_index(),
                                            factor_sets, new_custom_params,
                                            default_value)
                    if isinstance(results, tuple):
                        raw_fitness, self._retain_data = results
                    else:
                        raw_fitness = results

                    self._raw_fitness = default_value if np.isnan(
                        raw_fitness) else raw_fitness

                if self._raw_fitness == default_value:
                    self._is_valid = False

        except Exception as e:
            self._raw_fitness = default_value
            self._is_valid = False
            kd_logger.exception(e)

        self._final_fitness = self._raw_fitness

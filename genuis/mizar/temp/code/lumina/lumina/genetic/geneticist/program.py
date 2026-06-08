# -*- coding: utf-8 -*-
import pdb, json
import numpy as np
import time, datetime, hashlib, copy
from ultron.factor.genetic.geneticist.operators import crossover_sets, mutation_sets, calc_factor, Function, FunctionType
from ultron.factor.genetic.geneticist.program import Program as BaseProgram
from ultron.utilities.logger import kd_logger
import warnings

warnings.filterwarnings("ignore")

ABS_FLOAT = 0.000001


class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class Program(BaseProgram):

    def __init__(self,
                 init_depth,
                 method,
                 random_state,
                 factor_sets,
                 function_set,
                 operators_set,
                 signals_methods,
                 strategies_methods,
                 p_point_replace,
                 gen,
                 fitness,
                 coverage_rate=0.5,
                 n_features=0,
                 program=None,
                 parents=None,
                 operator_probs=None,
                 factor_probs=None,
                 signal_probs=None,
                 strategy_probs=None):
        self._init_depth = init_depth
        self._init_method = method
        self._program = program
        self._factor_sets = factor_sets
        self._p_point_replace = p_point_replace
        self._function_set = function_set
        self._operators_set = operators_set
        self._signals_methods = signals_methods
        self._strategies_methods = strategies_methods
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
        self._name = 'ultron_' + str(
            int(time.time() * 1000000 + datetime.datetime.now().microsecond))

        self._operator_probs = operator_probs
        self._factor_probs = factor_probs
        self._signal_probs = signal_probs
        self._strategy_probs = strategy_probs

        self._factor_data = None  ## 特征值
        self._signal_data = None  ## 信号值
        self._position_data = None  ## 目标仓位值

        self._temp_data = None

        if self._program is None:
            self._program, self._signal_method, self._strategy_method = self.build_program(
                random_state)
        else:
            ## 信号策略选择
            if self._signal_probs is None:
                self._signal_method = self._signals_methods[
                    random_state.randint(len(self._signals_methods))]
            else:
                self._signal_method = random_state.choice(
                    self._signals_methods, p=self._signal_probs)

            ## 交易策略选择
            if self._strategy_probs is None:
                self._strategy_method = self._strategies_methods[
                    random_state.randint(len(self._strategies_methods))]
            else:
                self._strategy_method = random_state.choice(
                    self._strategies_methods, p=self._strategy_probs)

        self.variation()
        self.create_identification()

    def penalty(self, penalty, max_corr, alpha):
        self._final_fitness -= penalty
        self._max_corr = max_corr
        self._penalty = penalty
        self._alpha = alpha

    @property
    def position_data(self):
        if self._temp_data is None:
            position_data = self._position_data
            position_data = position_data.reset_index().set_index(
                ['trade_time', 'code'])['transformed']
            position_data.name = self._name
            self._temp_data = position_data
        return self._temp_data

    def create_identification(self):
        m = hashlib.md5()
        try:
            token = self.transform()
        except Exception as e:
            #ID为key
            token = self._name
        if token is None:
            token = self._name
        ## 追加策略参数
        tokens = {
            'signal_method': self._signal_method.name,
            'signal_params': self._signal_method.params,
            'strategy_method': self._strategy_method.name,
            'strategy_params': self._strategy_method.params,
            'transform': token,
        }
        m.update(json.dumps(tokens, cls=NpEncoder).encode('utf-8'))
        self._identification = m.hexdigest()

    def output(self):

        parents = {'method': 'Gen'} if self._parents is None else self._parents
        return {
            'name': self._name,
            'method': parents['method'],
            'gen': self._gen,
            'features': self._identification,
            'formual': self.transform(),
            'raw_fitness': self._raw_fitness,
            'final_fitness': self._final_fitness,
            'strategy_method': self._strategy_method.name,
            'strategy_params': self._strategy_method.params,
            'signal_method': self._signal_method.name,
            'signal_params': self._signal_method.params,
            'alpha': self._alpha,
            'penalty': self._penalty,
            'max_corr': self._max_corr,
            'update_time': self._create_time
        }

    def build_program(self, random_state):
        start_time = time.time()
        #在范围内选取树形深度
        if self._init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self._init_method
        if isinstance(self._init_depth, int):
            max_depth = self._init_depth
        else:
            max_depth = random_state.randint(*self._init_depth)

        ## 特征算子选择
        if self._operator_probs is None:
            function = self._operators_set[random_state.randint(
                len(self._operators_set))]
        else:
            function = random_state.choice(self._operators_set,
                                           p=self._operator_probs)
        program = [function]
        terminal_stack = [function.arity]

        ## 信号策略选择
        if self._signal_probs is None:
            signal_method = self._signals_methods[random_state.randint(
                len(self._signals_methods))]
        else:
            signal_method = random_state.choice(self._signals_methods,
                                                p=self._signal_probs)

        ## 交易策略选择
        if self._strategy_probs is None:
            strategy_method = self._strategies_methods[random_state.randint(
                len(self._strategies_methods))]
        else:
            strategy_method = random_state.choice(self._strategies_methods,
                                                  p=self._strategy_probs)

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
                        return program, signal_method, strategy_method
                    terminal_stack[-1] -= 1
        kd_logger.info("init program cost time:{0}".format(time.time() -
                                                           start_time))
        return program, signal_method, strategy_method

    ##树变异
    def subtree_mutation(self, random_state):
        chicken, _, _ = self.build_program(random_state)
        return self.crossover(chicken, random_state)

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
                self._factor_data = factor_data.loc[factor_data.index.unique()
                                                    [backup_cycle:]]
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
                    factor_data1 = factor_data.reset_index().set_index(
                        ['trade_time', 'code'])
                    total_data1 = cycle_total_data.reset_index().set_index(
                        ['trade_time', 'code']).unstack()
                    signal_data = self._signal_method.function(
                        factor_data=factor_data1, **self._signal_method.params)
                    position_data = self._strategy_method.function(
                        signal=signal_data,
                        total_data=total_data1,
                        **self._strategy_method.params)

                    self._signal_data = signal_data.copy()
                    self._position_data = position_data.copy()

                    code = self._signal_data.columns[0]
                    self._signal_data.columns = ['transformed']
                    self._position_data.columns = ['transformed']
                    self._signal_data['code'] = code
                    self._position_data['code'] = code
                    ### 格式转化
                    results = self._fitness(
                        factor_data=self._factor_data,
                        pos_data=position_data,
                        total_data=total_data1,  #cycle_total_data.reset_index(),
                        signal_method=self._signal_method,
                        strategy_method=self._strategy_method,
                        factor_sets=factor_sets,
                        custom_params=new_custom_params,
                        default_value=default_value)
                    if isinstance(results, tuple):
                        raw_fitness, self._retain_data = results
                    else:
                        raw_fitness = results

                    self._raw_fitness = default_value if np.isnan(
                        raw_fitness) else raw_fitness
                else:
                    kd_logger.debug("{0} coverage {1} less {2}".format(
                        expression, coverage_rate, self._coverage_rate))
                if self._raw_fitness == default_value:
                    self._is_valid = False

        except Exception as e:
            kd_logger.error(
                "error=>desc:{0},exp:{1},signal_func:{2},signal_param:{3},stragegy_func:{4},stragegy_param:{5}"
                .format(e, expression, self._signal_method.name,
                        self._signal_method.params, self._strategy_method.name,
                        self._strategy_method.params))
            self._raw_fitness = default_value
            self._is_valid = False

        self._final_fitness = self._raw_fitness

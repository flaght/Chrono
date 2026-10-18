# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time, datetime, pickle, itertools, os, copy, pdb
from joblib import Parallel, delayed
from ultron.utilities.logger import kd_logger
from ultron.utilities.jobs import partition_estimators
from ultron.utilities.utils import check_random_state
from ultron.kdutils.progress import Progress
from ultron.factor.genetic.geneticist.operators import *
from lumina.genetic.signal.foundation import signals_methods
from lumina.genetic.strategy.foundation import strategies_methods
from lumina.genetic.geneticist.warehouse import WareHouse, callback_relevance
from lumina.genetic.geneticist.adaptive import Adaptive
from lumina.genetic.geneticist.thresholdor import Thresholdor
from .program import Program
#from .warehouse import sequential_gain

import warnings

warnings.filterwarnings("ignore")

MAX_INT = np.iinfo(np.int32).max
MIN_INT = np.iinfo(np.int32).min


def log_top_n_probs(description: str,
                    names: list,
                    probs: np.ndarray,
                    n: int = 5):
    """
    一个辅助函数，用于记录并打印概率最高的N个基因及其概率。

    :param description: str, 日志的描述性前缀 (e.g., "更新后算子概率").
    :param names: list, 包含所有基因名称的列表。
    :param probs: np.ndarray, 与names列表对应的概率数组。
    :param n: int, 要输出的最高概率的基因数量。
    """
    if not names or probs is None or len(names) != len(probs):
        kd_logger.warning(f"无法记录概率: {description} 的输入数据无效。")
        return
    # 1. 将名称和概率打包成元组列表
    combined = list(zip(names, probs))

    # 2. 根据概率（元组的第二个元素）进行降序排序
    #    使用 lambda 函数来指定排序的 key
    combined.sort(key=lambda x: x[1], reverse=True)

    # 3. 取出前N个
    top_n = combined[:n]

    # 4. 格式化输出字符串
    #    使用字典推导式和 f-string 来创建美观的输出
    formatted_output = {name: f"{prob:.2%}" for name, prob in top_n}

    # 5. 记录日志
    kd_logger.info(f"{description} (Top {n}): {formatted_output}")


def find_params(data_dict, keys, default=None):
    """
    更简洁的方式安全地从嵌套字典中获取值。
    """
    current_level = data_dict
    # 遍历到倒数第二个key，确保路径是通的
    for key in keys[:-1]:
        if not isinstance(current_level, dict):
            return default
        current_level = current_level.get(key, {})  # 如果中途断了，给一个空字典继续

    # 处理最后一个key
    if not isinstance(current_level, dict):
        return default
    return current_level.get(keys[-1], default)


def merge_positions(best_programs, is_concat=True):
    res = []
    for best_program in best_programs:
        position_data = best_program._position_data
        position_data = position_data.reset_index().set_index(
            ['trade_time', 'code'])['transformed'].sort_index()
        position_data.name = best_program._name
        res.append(position_data)
    return pd.concat(res, axis=1) if is_concat else res


def parallel_evolve(n_programs, parents, total_data, seeds, greater_is_better,
                    gen, params):
    tournament_size = params['tournament_size']
    function_set = params['function_set']
    operators_set = params['operators_set']
    signals_methods = params['signals_methods']
    strategies_methods = params['strategies_methods']
    init_depth = params['init_depth']
    init_method = params['init_method']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']
    factor_sets = params['factor_sets']
    fitness = params['fitness']
    backup_cycle = params['backup_cycle']
    custom_params = params['custom_params']

    operator_probs = params['operator_probs']
    factor_probs = params['factor_probs']
    signal_probs = params['signal_probs']
    strategy_probs = params['strategy_probs']

    def _contenders(tour_parents):
        contenders = random_state.randint(0, len(tour_parents), 2)
        return [tour_parents[p] for p in contenders]

    def _tournament(tour_parents):
        contenders = random_state.randint(0, len(tour_parents),
                                          tournament_size)
        raw_fitness = [tour_parents[p]._raw_fitness for p in contenders]
        if greater_is_better:
            parent_index = contenders[np.argmax(raw_fitness)]
        else:
            parent_index = contenders[np.argmin(raw_fitness)]
        return tour_parents[parent_index], parent_index

    programs = []
    with Progress(n_programs, 0, label='predict groups model') as pg:
        for i in range(n_programs):
            pg.show(i + 1)
            random_state = check_random_state(seeds[i])
            if parents is None:
                program = None
                genome = None
            else:
                method = random_state.uniform()
                parent, parent_index = _tournament(copy.deepcopy(parents))
                ori_parent = copy.deepcopy(parent)

                contenders = _contenders(copy.deepcopy(parents))
                for contender in contenders:
                    program, removed, remains = parent.crossover(
                        contender._program, random_state)
                    parent = Program(init_depth=init_depth,
                                     method=init_method,
                                     random_state=random_state,
                                     factor_sets=factor_sets,
                                     function_set=function_set,
                                     operators_set=operators_set,
                                     signals_methods=signals_methods,
                                     strategies_methods=strategies_methods,
                                     gen=gen,
                                     p_point_replace=p_point_replace,
                                     fitness=fitness,
                                     n_features=2,
                                     program=program,
                                     operator_probs=operator_probs,
                                     factor_probs=factor_probs,
                                     signal_probs=signal_probs,
                                     strategy_probs=strategy_probs)

                #新特征种群加入
                if random_state.uniform() < method_probs[2]:
                    program = Program(init_depth=init_depth,
                                      method=init_method,
                                      random_state=random_state,
                                      factor_sets=factor_sets,
                                      function_set=function_set,
                                      operators_set=operators_set,
                                      signals_methods=signals_methods,
                                      strategies_methods=strategies_methods,
                                      gen=gen,
                                      p_point_replace=p_point_replace,
                                      fitness=params['fitness'],
                                      n_features=2,
                                      program=None,
                                      operator_probs=operator_probs,
                                      factor_probs=factor_probs,
                                      signal_probs=signal_probs,
                                      strategy_probs=strategy_probs)
                    program, removed, remains = parent.crossover(
                        program._program, random_state)

                    parent = Program(init_depth=init_depth,
                                     method=init_method,
                                     random_state=random_state,
                                     factor_sets=factor_sets,
                                     function_set=function_set,
                                     operators_set=operators_set,
                                     signals_methods=signals_methods,
                                     strategies_methods=strategies_methods,
                                     gen=gen,
                                     p_point_replace=p_point_replace,
                                     fitness=params['fitness'],
                                     n_features=2,
                                     program=program,
                                     operator_probs=operator_probs,
                                     factor_probs=factor_probs,
                                     signal_probs=signal_probs,
                                     strategy_probs=strategy_probs)

                if method < method_probs[0]:  # # crossover
                    donor, donor_index = _tournament(copy.deepcopy(parents))
                    program, removed, remains = parent.crossover(
                        donor._program, random_state)
                    genome = {
                        'method': 'Crossover',
                        'parent_idx': parent_index,
                        'parent_nodes': removed,
                        'donor_idx': donor_index,
                        'donor_nodes': remains
                    }
                elif method < method_probs[1]:  # subtree_mutation
                    program, removed, _ = parent.subtree_mutation(random_state)
                    genome = {
                        'method': 'Subtree Mutation',
                        'parent_idx': parent_index,
                        'parent_nodes': removed
                    }
                elif method < method_probs[2]:  # hoist_mutation
                    program, removed = parent.hoist_mutation(random_state)
                    genome = {
                        'method': 'Hoist Mutation',
                        'parent_idx': parent_index,
                        'parent_nodes': removed
                    }
                elif method < method_probs[3]:  # point_mutation
                    program, mutated = parent.point_mutation(random_state)
                    genome = {
                        'method': 'Point Mutation',
                        'parent_idx': parent_index,
                        'parent_nodes': mutated
                    }
                else:
                    program = parent.reproduce()  # reproduction
                    genome = {
                        'method': 'Reproduction',
                        'parent_idx': parent_index,
                        'parent_nodes': []
                    }

                # 与原始自身进行交叉
                if random_state.uniform() < method_probs[3]:
                    program = Program(init_depth=init_depth,
                                      method=init_method,
                                      random_state=random_state,
                                      factor_sets=factor_sets,
                                      function_set=function_set,
                                      operators_set=operators_set,
                                      signals_methods=signals_methods,
                                      strategies_methods=strategies_methods,
                                      gen=gen,
                                      p_point_replace=p_point_replace,
                                      fitness=params['fitness'],
                                      n_features=2,
                                      program=program,
                                      parents=genome,
                                      operator_probs=operator_probs,
                                      factor_probs=factor_probs,
                                      signal_probs=signal_probs,
                                      strategy_probs=strategy_probs)
                    program, removed, remains = program.crossover(
                        ori_parent._program, random_state)

            program = Program(init_depth=init_depth,
                              method=init_method,
                              random_state=random_state,
                              factor_sets=factor_sets,
                              function_set=function_set,
                              operators_set=operators_set,
                              signals_methods=signals_methods,
                              strategies_methods=strategies_methods,
                              gen=gen,
                              p_point_replace=p_point_replace,
                              fitness=params['fitness'],
                              n_features=2,
                              program=program,
                              parents=genome,
                              operator_probs=operator_probs,
                              factor_probs=factor_probs,
                              signal_probs=signal_probs,
                              strategy_probs=strategy_probs)
            default_value = MIN_INT if greater_is_better else MAX_INT
            program.raw_fitness(total_data,
                                factor_sets,
                                default_value=default_value,
                                backup_cycle=backup_cycle,
                                custom_params=custom_params)
            programs.append(program)
        return programs


class Gentic(object):

    def __init__(
            self,
            population_size=2000,
            generations=MAX_INT,
            tournament_size=20,
            stopping_criteria=0.0,
            factor_sets=None,
            init_depth=(5, 6),
            init_method='full',
            operators_set=operators_sets,
            signals_methods=signals_methods,
            strategies_methods=strategies_methods,
            n_jobs=1,
            p_crossover=0.9,
            p_subtree_mutation=0.01,
            p_hoist_mutation=0.01,
            p_point_mutation=0.01,
            p_point_replace=0.05,
            greater_is_better=True,  #True 倒序， False 正序
            verbose=1,
            is_save=1,
            rootid=0,
            standard_score=2,  # None代表 根据tournament_size保留种群  standard_score保留种群
            out_dir='result',
            backup_cycle=0,  # 后备数据周期，主要用于在时间序列上的问题
            convergence=None,  # 收敛值，若为None，则不需要收敛值。
            low_memory=False,
            fitness=None,
            random_state=None,
            custom_params=None,
            save_model=None,
            relevance_penalty=None):
        self._population_size = population_size
        self._generations = MAX_INT if generations == 0 else generations
        self._tournament_size = tournament_size
        self._stopping_criteria = stopping_criteria
        self._factor_sets = factor_sets
        self._init_depth = init_depth
        self._init_method = init_method
        self._operators_set = operators_set
        self._function_set = [op.name for op in self._operators_set]
        self._signals_methods = signals_methods
        self._strategies_methods = strategies_methods
        self._p_crossover = p_crossover
        self._p_subtree_mutation = p_subtree_mutation
        self._p_hoist_mutation = p_hoist_mutation
        self._p_point_mutation = p_point_mutation
        self._p_point_replace = p_point_replace
        self._random_state = random_state
        self._greater_is_better = greater_is_better
        self._standard_score = standard_score
        self._fitness = fitness
        self._n_jobs = n_jobs
        self._backup_cycle = backup_cycle
        self._custom_params = custom_params
        self._low_memory = low_memory
        self._verbose = verbose
        self._is_save = is_save
        self._out_dir = out_dir
        self._convergence = convergence
        self._rootid = int(time.time() * 1000000 +
                           datetime.datetime.now().microsecond) if int(
                               rootid) == 0 else rootid
        self._save_model = self.save_model if save_model is None else save_model
        self._con_time = 0
        self._best_fitness = 0

        self._relevance_penalty = relevance_penalty if relevance_penalty is not None else callback_relevance
        # - 目标基准库大小为30
        # - 每新增20个策略就触发一次蒸馏
        self._ware_house = WareHouse(rootid=self._rootid,
                                     n_benchmark_clusters=find_params(
                                         custom_params,
                                         ['warehouse', 'initial_alpha'], 100),
                                     distill_trigger_size=find_params(
                                         custom_params,
                                         ['warehouse', 'distill_trigger_size'],
                                         20))

        ## 惩罚系数自适应
        self.adaptive_alpha = Adaptive(
            initial_alpha=find_params(custom_params,
                                      ['adaptive', 'initial_alpha'], 0.05),
            target_penalty_ratio=find_params(
                custom_params, ['adaptive', 'target_penalty_ratio'], 0.5),
            adjustment_speed=find_params(custom_params,
                                         ['adaptive', 'target_penalty_ratio'],
                                         0.1),
            lookback_period=find_params(custom_params,
                                        ['adaptive', 'lookback_period'], 5))
        ## 阈值自适应
        self.thresholdor = Thresholdor(
            initial_threshold=find_params(custom_params,
                                          ['threshold', 'initial_threshold'],
                                          1),
            target_percentile=find_params(custom_params,
                                          ['threshold', 'target_percentile'],
                                          0.65),
            min_threshold=find_params(custom_params,
                                      ['threshold', 'min_threshold'], 0.7),
            max_threshold=find_params(custom_params,
                                      ['threshold', 'max_threshold'], 4.0),
            adjustment_speed=find_params(custom_params,
                                         ['threshold', 'adjustment_speed'],
                                         0.1))
        self._operator_probs = None
        self._factor_probs = None
        self._strategy_probs = None
        self._signal_probs = None

    def save_model(self, gen, rootid, best_programs, custom_params):
        result_list = [{
            'transform': program.transform(),
            'fitness': program._raw_fitness
        } for program in best_programs]
        out_dir = os.path.join(self._out_dir, str(rootid))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        filename = os.path.join(out_dir, 'ultron_' + str(gen) + '.pkl')
        with open(filename, 'wb') as f:
            pickle.dump([result_list], f)

    def filter_programs(self, gen, population, standard_score=None):
        ## 保留符合条件的种群(1.种群有效 2.分数优于基准分 3.符合指定个数)
        standard_score = standard_score if standard_score is not None else self._standard_score * 0.7
        valid_prorams = np.array(population)[[
            program._is_valid for program in population
        ]]  # 只保留有效种群

        ## 删除重复种群
        identification_dict = {}
        for program in valid_prorams:
            identification_dict[program._identification] = program

        valid_prorams = list(identification_dict.values())
        fitness = [program._final_fitness for program in valid_prorams]
        if standard_score is not None:  #分数筛选且第二代开始
            if self._greater_is_better:
                best_programs = np.array([
                    program for program in valid_prorams
                    if program._final_fitness >= standard_score
                ])
            else:
                best_programs = np.array([
                    program for program in valid_prorams
                    if program._final_fitness <= standard_score
                ])

        #若不满足分数，则进行排序选出前_tournament_size
        if len(best_programs
               ) < self._tournament_size or self._standard_score is None:
            if self._greater_is_better:
                best_programs = np.array(valid_prorams)[np.argsort(
                    fitness)[-self._tournament_size:]]
            else:
                best_programs = np.array(valid_prorams)[np.argsort(
                    fitness)[:self._tournament_size]]
        return best_programs

    def train(self, total_data):
        random_state = check_random_state(self._random_state)
        self._method_probs = np.array([
            self._p_crossover, self._p_subtree_mutation,
            self._p_hoist_mutation, self._p_point_mutation
        ])

        self._method_probs = np.cumsum(self._method_probs)

        if self._method_probs[-1] > 1:
            raise ValueError('The sum of p_crossover, p_subtree_mutation, '
                             'p_hoist_mutation and p_point_mutation should '
                             'total to 1.0 or less.')

        if self._init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include '
                             '"grow", "full" and "half and half". Given %s.' %
                             self._init_method)

        if (isinstance(self._init_depth, tuple)
                and len(self._init_depth) != 2):
            raise ValueError('init_depth should be a tuple with length two.')

        if (isinstance(self._init_depth, tuple)
                and (self._init_depth[0] > self._init_depth[1])):
            raise ValueError('init_depth should be in increasing numerical '
                             'order: (min_depth, max_depth).')

        params = {}
        params['tournament_size'] = self._tournament_size
        params['function_set'] = self._function_set
        params['operators_set'] = self._operators_set
        params['signals_methods'] = self._signals_methods
        params['strategies_methods'] = self._strategies_methods
        params['init_depth'] = self._init_depth
        params['init_method'] = self._init_method
        params['method_probs'] = self._method_probs
        params['p_point_replace'] = self._p_point_replace
        params['factor_sets'] = self._factor_sets
        params['fitness'] = self._fitness
        params['backup_cycle'] = self._backup_cycle
        params['custom_params'] = self._custom_params
        self._programs = []
        self._best_programs = None
        self._run_details = {
            'generation': [],
            'average_fitness': [],
            'best_fitness': [],
            'generation_time': [],
            'best_programs': []
        }

        prior_generations = len(self._programs)
        n_more_generations = self._generations - prior_generations
        for gen in range(prior_generations, self._generations):
            start_time = time.time()

            kd_logger.info("start {0}/{1} generations ".format(
                gen, self._generations))
            if gen == 0:
                parents = None
            else:
                parents = self._programs[gen - 1]
                parents = [parent for parent in parents if parent._is_valid]
                kd_logger.info("提取上一代有效种群:{0}".format(len(parents)))

                ## 更新 概率
                parents = [(setattr(p, '_operator_probs',
                                    self._operator_probs),
                            setattr(p, '_factor_probs', self._factor_probs),
                            setattr(p, '_signal_probs', self._signal_probs),
                            setattr(p, '_strategy_probs',
                                    self._strategy_probs), p)[-1]
                           for p in parents]

            n_jobs, n_programs, starts = partition_estimators(
                self._population_size, self._n_jobs)

            seeds = random_state.randint(MAX_INT, size=self._population_size)
            ## 动态更新概率
            params['operator_probs'] = self._operator_probs
            params['factor_probs'] = self._factor_probs
            params['signal_probs'] = self._signal_probs
            params['strategy_probs'] = self._strategy_probs

            kd_logger.debug(
                f"""update probs  operator: {params['operator_probs']} \n factor {params['factor_probs']}\n signal:{params['signal_probs']}\n trategy:{params['strategy_probs']}\n\n"""
            )

            population = Parallel(n_jobs=n_jobs, verbose=self._verbose)(
                delayed(
                    parallel_evolve)(n_programs[i], parents, total_data, seeds,
                                     self._greater_is_better, gen, params)
                for i in range(n_jobs))

            population = list(itertools.chain.from_iterable(population))
            #剔除无效因子
            valid_count = len(population)
            population = [
                program for program in population if program._is_valid
            ]

            kd_logger.info("1. 剔除无效program {}-->{}，保留率:{}".format(
                valid_count, len(population),
                len(population) / float(valid_count)))

            if len(population) == 0:
                break

            # 保留当前最有效的种群 用于下一代
            '''
            if self._best_programs is None:  ## 判断是否是第一代
                self._programs.append(population)
            else:
                identification_dict = {}
                valid_prorams = list(
                    np.concatenate([population, self._best_programs]))
                for program in valid_prorams:
                    identification_dict[program._identification] = program
                valid_prorams = list(identification_dict.values())
                self._programs.append(valid_prorams)
            '''

            # ===== 2. 更新并获取动态筛选阈值 =====
            # 收集当前种群的 fitness 列表
            current_population_fitness = [
                p._final_fitness for p in population
                if p._final_fitness is not None
            ]

            if 'gain' in self._custom_params:
                # 更新阈值调节器
                self.thresholdor.update(current_population_fitness)
                # 更新分数--> 筛选分数，包括绝对筛选和相关性筛选
                self._standard_score = self.thresholdor.threshold()
                self._custom_params['gain'][
                    'fitness_threshold'] = self._custom_params['gain'][
                        'fitness_scale'] * self._standard_score

                kd_logger.info(
                    f"阈值分数刷新:fitness_threshold:{self._custom_params['gain']['fitness_threshold']}, standard_score:{self._standard_score}"
                )
            else:
                kd_logger.info("未配置动态阈值更新")

            ## 过滤一次，原始分数达不到阈值，惩罚后分数更达不到。降低相关性惩罚计算(慢) 影响时间
            filter_count = len(population)
            population = self.filter_programs(gen, population, 0)  # 大于0即可
            kd_logger.info("2.低分过滤部分无效:{}--{} 保留率:{}".format(
                filter_count, len(population),
                len(population) / float(filter_count)))

            ## 回调函数相关性惩罚 调整fitness分数
            if self._ware_house.permanent_core is not None:
                if 'warehouse' in self._custom_params:
                    population = self._relevance_penalty(
                        best_programs=population,  ## 对整个 population操作
                        benchmark_warehouse=self._ware_house.
                        benchmark_warehouse,
                        alpha=self.adaptive_alpha.alpha)
                else:
                    kd_logger.info("没有配置与基础库相关性惩罚")

            last_count = len(population)
            ## 先加入上代精英，上代精英中可能会有为了满足数量而不满足质量
            if self._best_programs is not None:
                # 将上一代的精英也加入进来
                merged_population = list(
                    np.concatenate([population, self._best_programs]))
                # 去重
                temp_dict = {p._identification: p for p in merged_population}
                merged_population = list(temp_dict.values())
            else:
                merged_population = population

            kd_logger.info("3.与上一代合并:{}--{} 新增率:{}".format(
                last_count, len(merged_population),
                (len(merged_population) - last_count) / last_count))

            filter_count = len(merged_population)
            best_programs = self.filter_programs(gen, merged_population,
                                                 0)  #继续使用大于0

            kd_logger.info("4.标准过滤 {}/{} 保留比例:{}".format(
                len(best_programs), filter_count,
                len(best_programs) / float(filter_count)))

            ##
            #减慢进化速度，并且过早地剔除了多样性。
            '''
            selected_poistions = sequential_gain(
                best_programs,
                total_data=total_data,
                strategy_settings=self._custom_params['strategy_settings'],
                corr_threshold=find_params(self._custom_params,
                                           ['gain', 'corr_threshold'], 0.5),
                fitness_threshold=find_params(self._custom_params,
                                              ['gain', 'fitness_threshold'],
                                              0.8),
                gain_threshold=find_params(self._custom_params,
                                           ['gain', 'gain_threshold'], 0.2))
            
            best_programs = [
                program for program in best_programs
                if program._name in selected_poistions.columns
            ]
            '''
            self._best_programs = best_programs

            ## 更新基准库
            if 'warehouse' in self._custom_params:
                if self._ware_house.permanent_core is None:
                    candidate_positions = merge_positions(
                        self._best_programs, True)
                    core_positions = candidate_positions
                    self._ware_house.set_initial_benchmark(
                        core_positions=core_positions)
                else:
                    candidate_positions = merge_positions(
                        self._best_programs, False)
                    for candidate_position in candidate_positions:
                        self._ware_house.add_new_position(candidate_position)
            else:
                kd_logger.info("没有配置与基础库相关性惩罚")

            fitness = [
                program._final_fitness for program in self._best_programs
                if not np.isnan(program._final_fitness)
            ]
            self._run_details['generation'].append(gen)
            self._run_details['average_fitness'].append(np.mean(fitness))
            generation_time = time.time() - start_time
            self._run_details['generation_time'].append(generation_time)
            self._run_details['best_programs'].append(self._best_programs)
            kd_logger.info(
                'ExpendTime:%f,Generation:%d,Tournament:%d, Fitness Mean:%f,Fitness Max:%f,Fitness Min:%f'
                % (generation_time, gen, len(best_programs), np.mean(fitness),
                   np.max(fitness), np.min(fitness)))

            ## 动态更新基因概率
            self._operator_probs, self._factor_probs, self._signal_probs, self._strategy_probs = self.probs(
                best_programs=self._best_programs,
                operators_set=self._operators_set,
                factors_sets=self._factor_sets,
                signals_sets=signals_methods,
                strategy_sets=strategies_methods)

            ## 刷新自适应度 找到最好的原始分和对应的最大惩罚度
            if len(self._best_programs
                   ) > 0 and 'adaptive' in self._custom_params:
                best_program = max(self._best_programs,
                                   key=lambda p: p._raw_fitness)
                if best_program._raw_fitness is not None and best_program._max_corr != 0:
                    self.adaptive_alpha.update(
                        base_performance=best_program._raw_fitness,
                        max_corr=best_program._max_corr)

            #保存每代信息
            if self._is_save:
                self._save_model(gen, self._rootid,
                                 self._run_details['best_programs'][-1],
                                 self._custom_params, total_data)

            ## 下一代返回准备
            self._programs.append(self._best_programs)
            kd_logger.info("添加有效种群:{0}".format(len(self._best_programs)))

            if self._greater_is_better:
                best_fitness = fitness[np.argmax(fitness)]
                if best_fitness >= self._stopping_criteria:
                    break
            else:
                best_fitness = fitness[np.argmin(fitness)]
                if best_fitness <= self._stopping_criteria:
                    break

            if np.mean(
                    fitness
            ) == MIN_INT or best_fitness == MIN_INT or best_fitness == np.nan:
                break
            self._run_details['best_fitness'].append(best_fitness)
            # 收敛值判断
            if self._convergence is None or gen == 0:
                continue
            d_value = np.mean(fitness) - self._run_details['average_fitness'][
                gen - 1]
            kd_logger.debug('d_value:%f,convergence:%f,con_time:%d' %
                            (d_value, self._convergence, self._con_time))
            if abs(d_value) < self._convergence:
                self._con_time += 1
                if self._con_time > 5:
                    break
            else:
                self._con_time = 0

    def probs(self, best_programs, operators_set, factors_sets, signals_sets,
              strategy_sets):
        op_counts = {op.name: 0 for op in operators_set}
        factor_counts = {term: 0 for term in factors_sets}
        signal_counts = {sg.name: 0 for sg in signals_sets}
        strategy_counts = {sy.name: 0 for sy in strategy_sets}

        for best_program in best_programs:
            for node in best_program._program:
                if isinstance(node, Function):
                    op_counts[node.name] += 1
                elif node in factor_counts:
                    factor_counts[node] += 1

            signal_counts[best_program._signal_method.name] += 1
            strategy_counts[best_program._strategy_method.name] += 1

        op_values = np.array(
            [op_counts.get(op.name, 0) + 1 for op in operators_set])
        factor_values = np.array(
            [factor_counts.get(factor, 0) + 1 for factor in factors_sets])
        signal_values = np.array(
            [signal_counts.get(signal.name, 0) + 1 for signal in signals_sets])
        strategy_values = np.array([
            strategy_counts.get(strategy.name, 0) + 1
            for strategy in strategy_sets
        ])

        if 'probability' not in self._custom_params or 'method' not in self._custom_params[
                'probability']:
            method = 'equal'
        else:
            method = 'weight'
        op_probs = op_values / op_values.sum(
        ) if method == 'weight' else np.full(len(op_values), 1.0 /
                                             len(op_values))
        factor_probs = factor_values / factor_values.sum(
        ) if method == 'weight' else np.full(len(factor_values), 1.0 /
                                             len(factor_values))
        signal_probs = signal_values / signal_values.sum(
        ) if method == 'weight' else np.full(len(signal_values), 1.0 /
                                             len(signal_values))
        strategy_probs = strategy_values / strategy_values.sum(
        ) if method == 'weight' else np.full(len(strategy_values), 1.0 /
                                             len(strategy_values))

        op_names = [op.name for op in operators_set]
        factor_names = list(factors_sets)  # factors_sets 本身就是列表
        # 对于信号和策略，我们可能想看到更详细的名称（包含参数）
        signal_names = [s.name for s in signals_sets]
        strategy_names = [s.name for s in strategy_sets]

        log_top_n_probs("更新后算子概率", op_names, op_probs)
        log_top_n_probs("更新后特征概率", factor_names, factor_probs)
        log_top_n_probs("更新后信号概率", signal_names, signal_probs)
        log_top_n_probs("更新后策略概率", strategy_names, strategy_probs)
        '''
        kd_logger.info(
            f"更新后算子概率 (部分): "
            f"{ {k.name: v for k, v in list(zip(operators_set, op_probs))[:5]} }"
        )
        kd_logger.info(
            f"更新后特征概率 (部分): "
            f"{ {k: v for k, v in list(zip(factors_sets, factor_probs))[:5]} }"
        )
        kd_logger.info(
            f"更新后信号概率 (部分): "
            f"{ {k.name: v for k, v in list(zip(signals_sets, signal_probs))[:5]} }"
        )

        kd_logger.info(
            f"更新后策略概率 (部分): "
            f"{ {k.name: v for k, v in list(zip(strategy_sets, strategy_probs))[:5]} }"
        )
        '''
        return op_probs, factor_probs, signal_probs, strategy_probs

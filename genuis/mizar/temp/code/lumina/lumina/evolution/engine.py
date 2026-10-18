import numpy as np
import time, datetime, pickle, itertools, os, pdb
from lumina.evolution.genetic import Gentic
from ultron.utilities.logger import kd_logger
from ultron.factor.genetic.geneticist.operators import operators_sets

MAX_INT = np.iinfo(np.int32).max
MIN_INT = np.iinfo(np.int32).min


class Engine(object):

    def __init__(
            self,
            population_size=2000,
            generations=MAX_INT,
            tournament_size=20,
            stopping_criteria=0.0,
            factor_sets=None,
            init_depth=(5, 6),
            init_method='full',
            operators_sets=operators_sets,
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
            relevance_penalty=None):  #相关性惩罚
        self._population_size = population_size
        self._tournament_size = tournament_size
        self._generations = generations
        self._init_depth = init_depth
        self._n_jobs = n_jobs
        self._rootid = rootid
        self._init_method = init_method
        self._stopping_criteria = stopping_criteria
        self._p_crossover = p_crossover
        self._p_subtree_mutation = p_subtree_mutation
        self._p_hoist_mutation = p_hoist_mutation
        self._p_point_mutation = p_point_mutation
        self._p_point_replace = p_point_replace
        self._factor_sets = factor_sets
        self._standard_score = standard_score
        self._operators_sets = operators_sets
        self._convergence = convergence
        self._fitness = fitness
        self._save_model = save_model
        self._custom_params = custom_params
        self._backup_cycle = backup_cycle
        self._relevance_penalty = relevance_penalty

    def run_gentic(self, total_data):
        kd_logger.info("开始重构新的挖掘")
        gentic = Gentic(population_size=self._population_size,
                        generations=self._generations,
                        tournament_size=self._tournament_size,
                        init_depth=self._init_depth,
                        n_jobs=self._n_jobs,
                        stopping_criteria=self._stopping_criteria,
                        p_crossover=self._p_crossover,
                        p_point_mutation=self._p_subtree_mutation,
                        p_subtree_mutation=self._p_subtree_mutation,
                        p_hoist_mutation=self._p_hoist_mutation,
                        p_point_replace=self._p_point_replace,
                        rootid=self._rootid,
                        factor_sets=self._factor_sets,
                        standard_score=self._standard_score,
                        operators_set=self._operators_sets,
                        convergence=self._convergence,
                        backup_cycle=self._backup_cycle,
                        fitness=self._fitness,
                        save_model=self._save_model,
                        custom_params=self._custom_params,
                        relevance_penalty=self._relevance_penalty)
        
        gentic.train(total_data=total_data)
        result = gentic._run_details
        raw_fitness = 0 if len(
            result['best_programs']) == 0 else result['best_fitness'][-1]
        del gentic
        return raw_fitness



    def train(self, total_data):
        raw_fitness = 0
        while raw_fitness < self._stopping_criteria:
            raw_fitness = self.run_gentic(total_data)

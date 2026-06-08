import time, pdb
from ultron.factor.genetic.geneticist.operators import custom_transformer
from lumina.genetic.signal.foundation import signals_methods
from lumina.genetic.strategy.foundation import strategies_methods
from lumina.genetic.geneticist.engine import Engine

## 挖掘策略
class Motor(object):

    def __init__(self, factor_columns, callback_fitness, callback_save_model):
        self._factor_columns = factor_columns
        self._callback_fitness = callback_fitness
        self._callback_save_model = callback_save_model

    ## 挖掘相关配置
    def gentic_configure(self, configure):

        def init_config(name, value, configure):
            configure[name] = configure[name] if name in configure else value

        init_config('population_size', 100, configure)  # 初始化种群数
        init_config('tournament_size', 20, configure)  # 每一代优秀种群数
        init_config('init_depth', 4, configure)  # 每个种群
        init_config('generations', 30, configure)  # 繁衍代数据
        init_config('n_jobs', 4, configure)  # 并发数
        init_config('stopping_criteria', 100, configure)  # 停止繁衍值即目标值大于预设值停止繁衍
        init_config('standard_score', 10, configure)  # 每一代保留优秀种群的预设值
        init_config('crossover', 0.4, configure)  # 交叉率
        init_config('point_mutation', 0.3, configure)  # 点变异率
        init_config('subtree_mutation', 0.1, configure)  # 树变异率
        init_config('hoist_mutation', 0.1, configure)  # 突变异率
        init_config('point_replace', 0.1, configure)  # 点交换率
        init_config('rootid', int(time.time()), configure)  # 节点
        init_config('convergence', 0.002, configure)  # 每一代收敛预设停止值

    def create_gentic(self, operators_sets, signals_methods,
                      strategies_methods, configure):
        engine = Engine(population_size=configure['population_size'],
                        tournament_size=configure['tournament_size'],
                        init_depth=(1, configure['init_depth']),
                        generations=configure['generations'],
                        n_jobs=configure['n_jobs'],
                        stopping_criteria=configure['stopping_criteria'],
                        p_crossover=configure['crossover'],
                        p_point_mutation=configure['point_mutation'],
                        p_subtree_mutation=configure['subtree_mutation'],
                        p_hoist_mutation=configure['hoist_mutation'],
                        p_point_replace=configure['point_replace'],
                        rootid=configure['rootid'],
                        factor_sets=self._factor_columns,
                        standard_score=configure['standard_score'],
                        operators_set=operators_sets,
                        signals_methods=signals_methods,
                        strategies_methods=strategies_methods,
                        backup_cycle=1,
                        convergence=configure['convergence'],
                        fitness=self._callback_fitness,
                        save_model=self._callback_save_model,
                        custom_params=configure['custom_params'])
        return engine

    def calculate(self,
                  total_data,
                  configure,
                  operators_sets,
                  signals_sets=None,
                  strategies_sets=None,
                  custom_params=None):

        ## 算子处理
        operators_sets = custom_transformer(operators_sets)
        ## 信号函数
        signals_methods1 = signals_sets if isinstance(
            signals_sets, list) else signals_methods
        ## 策略函数
        strategies_methods1 = strategies_sets if isinstance(
            strategies_sets, list) else strategies_methods
        ## 处理配置
        self.gentic_configure(configure)

        gentic = self.create_gentic(operators_sets, signals_methods1,
                                    strategies_methods1, configure)

        gentic.train(total_data=total_data)

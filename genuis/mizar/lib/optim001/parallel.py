import pandas as pd
import pdb,itertools
from typing import List, Dict, Any
from ultron.ump.core.process import add_process_env_sig, EnvProcess
from ultron.kdutils.parallel import delayed, Parallel
from kdutils.process import split_k, run_process, create_parellel
from lib.optim001.optimizer import FactorOptimizer, create_name_id


def optimize_factors(column, n_trials, top_n, period, verbose, multi_objective,
                     operators_pd, fields_pd, total_data, total_data1,
                     optimize_rule, objective_func):
    res = []
    optimizer = FactorOptimizer(operators_pd=operators_pd, fields_pd=fields_pd)
    objectives = objective_func(expression=column,
                                period=period,
                                total_data=total_data,
                                total_data1=total_data1,
                                optimize_rule=optimize_rule)
    res.append({
        'name': "ultron_{0}".format(create_name_id(column)),
        "formual": column,
        "final_fitness": objectives[0]
    })
    
    results = optimizer.optimize_expression(
        expression=column,
        objective_function=objective_func,
        n_trials=n_trials,  # 多目标优化需要更多试验
        total_data=total_data,
        total_data1=total_data1,
        period=period,
        optimize_parameters=True,
        optimize_operators=True,
        optimize_fields=True,
        optimize_rule=optimize_rule,
        study_name=f"multi_objective_{column}",
        multi_objective=multi_objective,  # 启用多目标优化
        verbose=verbose,
        top_n=top_n  
    )

    res += [{
        "name": "ultron_{0}".format(create_name_id(result['expression'])),
        'formual': result['expression'],
        'final_fitness': result['score'][0]
    } for result in results['top_n_results']]
    final_programs = pd.DataFrame(res)
    return final_programs


@add_process_env_sig
def run_optimizer(target_column, n_trials, top_n, period, multi_objective,
                  verbose, operators_pd, fields_pd, total_data, total_data1,
                  optimize_rule, objective_func):
    batch_final_programs = run_process(target_column=target_column,
                                       callback=optimize_factors,
                                       n_trials=n_trials,
                                       top_n=top_n,
                                       period=period,
                                       multi_objective=multi_objective,
                                       verbose=verbose,
                                       operators_pd=operators_pd,
                                       fields_pd=fields_pd,
                                       total_data=total_data,
                                       total_data1=total_data1,
                                       optimize_rule=optimize_rule,
                                       objective_func=objective_func)
    return batch_final_programs


class ParallelOptimizer:

    def __init__(self,
                 operators_pd: pd.DataFrame,
                 fields_pd: pd.DataFrame,
                 n_jobs: int = 1):
        """
        初始化优化器
        
        Args:
            operators_pd: 算子依赖关系
            fields_pd: 字段依赖关系
            n_jobs: 并行进程数 (1=单进程, >1=多进程)
        """
        self.operators_pd = operators_pd
        self.fields_pd = fields_pd
        self.n_jobs = max(1, n_jobs)  # 至少1个进程

    def optimize(self,
                 expressions: List[str],
                 objective_function: callable,
                 total_data: pd.DataFrame,
                 total_data1: pd.DataFrame,
                 period: int,
                 optimize_rule: Dict,
                 n_trials: int = 100,
                 multi_objective: bool = False,
                 top_n: int = 1,
                 verbose: bool = True):

        process_list = split_k(self.n_jobs, expressions)
        res = create_parellel(process_list=process_list,
                              callback=run_optimizer,
                              n_trials=n_trials,
                              top_n=top_n,
                              period=period,
                              verbose=verbose,
                              multi_objective=multi_objective,
                              operators_pd=self.operators_pd,
                              fields_pd=self.fields_pd,
                              total_data=total_data,
                              total_data1=total_data1,
                              optimize_rule=optimize_rule,
                              objective_func=objective_function)
        pdb.set_trace()
        res = list(itertools.chain.from_iterable(res))
        results =  pd.concat(res,axis=0)
        result = results.drop_duplicates(subset=['name'])
        return result

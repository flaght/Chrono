import os,optuna,pdb,itertools
import pandas as pd
from typing import List, Dict, Any, Tuple
from optuna.trial import TrialState
from ultron.ump.core.process import add_process_env_sig, EnvProcess
from kdutils.process import split_k, run_process, create_parellel
from lib.optim002.calculator import ImpulseCalculator
from lib.cux001 import FactorEvaluate1


def optimize_factor(column: str, market_data: pd.DataFrame,
                     returns_data: pd.DataFrame,
                     n_trials: int, timeout: int,
                     period: int,
                     top_n: int,
                     impulse_version: str,
                     evaluator_params: Dict,
                     optimize_rule:Dict,
                     param_ranges: Dict[str, Dict] = None):
    ## 优化单个因子的函数 - 参照ParallelOptimizer模式
    res = []
    try:
        impulse_calculator = ImpulseCalculator(impulse_version)
        optimizer = ImpulseParameterOptimizer(impulse_calculator=impulse_calculator,
                        evaluator_params=evaluator_params,
                        param_ranges=param_ranges)
        result = optimizer.optimize(impulse_calculator.get_class(column), market_data=market_data,
                       returns_data=returns_data, period=period,
                       optimize_rule=optimize_rule,
                       n_trials=n_trials, timeout=timeout, top_n=top_n) 
        ### 保存绩效符合的 然后转成dataframe
        if result and not result.get('error', False):
            for  trial in result['all_trials']:
                res.append({
                    'factor_name': column,
                    'params': trial['params'],
                    'ic_mean':trial['values'][0],
                    'sharpe2':trial['values'][1],
                    'calmar':trial['values'][2]
                    }
                )
    except Exception as e:
        print(f"Error optimizing {column}: {e}")
    return pd.DataFrame(res)

        

@add_process_env_sig
def run_optimization(target_column: str, market_data: pd.DataFrame,
                    returns_data: pd.DataFrame,
                    period:int,
                    n_trials: int, timeout: int, top_n:int,
                    impulse_version: str,
                    evaluator_params: Dict,
                    optimize_rule:Dict,
                    param_ranges: Dict[str, Dict] = None):
    batch_results = run_process(target_column=target_column,
                               callback=optimize_factor,
                               market_data=market_data,
                               returns_data=returns_data,
                               period=period,
                               n_trials=n_trials,
                               timeout=timeout,
                               top_n=top_n,
                               optimize_rule=optimize_rule,
                               impulse_version=impulse_version,
                               evaluator_params=evaluator_params,
                               param_ranges=param_ranges)
    return batch_results


class ImpulseParameterOptimizer:

    def __init__(self, impulse_calculator: ImpulseCalculator, evaluator_params: Dict, param_ranges: Dict[str, Dict] = None):
        """
        初始化参数优化器

        Args:
            impulse_calculator: Impulse计算器实例
            evaluator_template: FactorEvaluate1的配置模板
            param_ranges: 自定义参数搜索空间范围，为None时使用默认值
        """
        self.calculator = impulse_calculator
        self.evaluator_params = evaluator_params
        self.param_ranges = param_ranges

    def get_dynamic_param_ranges(self, factor_class) -> Dict[str, Dict]:
        """
        根据因子参数格式动态生成搜索空间
        """
        param_format = self.calculator.detect_parameter_format(factor_class)

        # 基于参数格式调整范围
        ranges = {}
        for param_name in param_format['param_names']:
            if param_name in self.param_ranges:
                ranges[param_name] = self.param_ranges[param_name]
            else:
                # 默认范围
                ranges[param_name] = {'min': 3, 'max': 50, 'step': 1}

        return ranges


    def _objective(self, trial: optuna.Trial, factor_class, market_data: pd.DataFrame,
                  returns_data: pd.DataFrame, period: int,
                  evaluator_params: Dict,
                  optimize_rule:Dict,
                  param_ranges: Dict[str, Dict] = None) -> Tuple[float]:
        # 如果没有传入param_ranges，使用动态生成的范围
        if param_ranges is None:
            param_ranges = self.get_dynamic_param_ranges(factor_class)

        # 检测因子参数格式 - 每个因子可能有多个参数
        param_format = self.calculator.detect_parameter_format(factor_class)

        # 为该因子的每个参数采样优化值
        # 例如: ImpulseKx001有['window', 'weriod', 'ewm'] 3个参数
        #      ImpulseKx005有['window', 'fast', 'slow', 'weriod', 'ewm'] 5个参数
        sampled_params = {}
        for param_name in param_format['param_names']:  # 遍历该因子的所有参数
            if param_name in param_ranges:
                range_config = param_ranges[param_name]
                if 'choices' in range_config:  # 分类参数 (如ewm: 0或1)
                    sampled_params[param_name] = trial.suggest_categorical(
                        param_name, range_config['choices'])
                else:  # 连续整数参数 (如window: 3-100)
                    sampled_params[param_name] = trial.suggest_int(
                            param_name,
                            range_config.get('min', 3),
                            range_config.get('max', 50),
                            step=range_config.get('step', 1))
            else:
                # 参数不在自定义范围内，使用默认值
                if param_name == 'ewm':  # ewm必须是分类参数
                    sampled_params[param_name] = trial.suggest_categorical(param_name, [0, 1])
                else:  # 其他参数使用连续整数默认值
                    sampled_params[param_name] = trial.suggest_int(param_name, 3, 50, step=1)

        
        params = tuple(sampled_params[param_name] for param_name in param_format['param_names'])

        # 计算因子
        result_dict = self.calculator.calculate_with_class(factor_class, [params], market_data)
        factor_name = list(result_dict.keys())[0]
        factor_values = result_dict[factor_name]
        factor_data = factor_values.to_frame('transformed')
        factor_data = factor_data.reset_index()
        # 评估因子
        ret_col = f'nxt1_ret_{period}h'
        eval_data = factor_data.merge(
            returns_data[['trade_time', 'code', ret_col]],
            on=['trade_time', 'code'],
            how='inner')
        
        evaluator_params = self.evaluator_params.copy()
        evaluator_params.update({
                'factor_data': eval_data,
                'factor_name': 'transformed',
                'ret_name': ret_col,
                'expression': factor_values.name,
                'name': factor_values.name,
                'resampling_win': period
            })
        
        evaluator = FactorEvaluate1(**evaluator_params)
        states = evaluator.run(is_check=False)


        # 一键获取所有优化目标的值
        # optimize_rule格式如: {'ic_mean': 'maximize', 'sharpe2': 'maximize', 'calmar': 'maximize'}
        # 从states字典中一次性提取所有目标指标的值
        target_values = [states.get(key, -1.0) for key in optimize_rule.keys()]

        # 对需要取绝对值的指标进行处理
        if 'ic_mean' in optimize_rule:
            target_values[list(optimize_rule.keys()).index('ic_mean')] = abs(target_values[list(optimize_rule.keys()).index('ic_mean')])

        # 返回多目标优化结果元组
        return tuple(target_values)
            

    def optimize(self, factor_class, market_data: pd.DataFrame,
                       returns_data: pd.DataFrame, 
                       optimize_rule:Dict,
                       period: int = 1,
                       n_trials: int = 100, timeout: int = 3600,
                       top_n:int = 10) -> Dict[str, Any]:
        # 创建多目标优化study
        study = optuna.create_study(
            directions=optimize_rule.values(),
            study_name=f"{factor_class.__name__}_optimization",
            sampler=optuna.samplers.NSGAIISampler()  # 多目标优化采样器
        )

        study.optimize(
            lambda trial: self._objective(trial=trial, factor_class=factor_class, 
                                    market_data=market_data,
                                    returns_data=returns_data, 
                                    optimize_rule=optimize_rule,
                                    evaluator_params=self.evaluator_params,
                                    param_ranges=self.param_ranges,
                                    period=period),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        # 检查所有完成的trial
        all_trials = study.trials
        best_trials = study.best_trials

        # 调试信息
        #print(f"总trial数: {len(all_trials)}, Pareto最优trial数: {len(best_trials)}")
        completed_trials = [t for t in all_trials if t.state == TrialState.COMPLETE]
        #print(f"完成trial数: {len(completed_trials)}")

        results = []
        # 如果best_trials为空，使用所有完成的trial
        if len(best_trials) == 0:
            best_trials = completed_trials[:top_n] if top_n > 0 else completed_trials
        else:
            best_trials = best_trials if top_n == 0 else best_trials[:top_n]
        
        for trial in best_trials:
            results.append({
                'params': trial.params,
                'values': trial.values,
                'number': trial.number
            })

        best_result = {
            'factor_name': factor_class.__name__,
            'best_params': best_trials[0].params,
            'best_values': best_trials[0].values,
            'all_trials': results,
            'n_trials_completed': len(study.trials)
        }   
        return best_result

        

class FactorsOptimizer(object):
    def __init__(self, impulse_version: str = 'i017', 
                 n_jobs: int = 4, 
                 evaluator_params: Dict = None, 
                 param_ranges: Dict[str, Dict] = None):
        self.impulse_version = impulse_version
        self.n_jobs = min(n_jobs, os.cpu_count() or 4)  
        self.evaluator_params = evaluator_params
        self.param_ranges = param_ranges

    def optimize_parallel(self, factor_names:List[str],
                                    market_data: pd.DataFrame,
                                    returns_data: pd.DataFrame,
                                    optimize_rule:Dict,
                                    period: int = 1, 
                                    n_trials: int = 100,
                                    top_n = 100,
                                    timeout: int = 3600) -> pd.DataFrame:

        factor_list = list(factor_names)
        process_list = split_k(self.n_jobs, factor_list)

        res = create_parellel(process_list=process_list,
                             callback=run_optimization,
                             market_data=market_data,
                             returns_data=returns_data,
                             period=period,
                             n_trials=n_trials,
                             timeout=timeout,
                             top_n=top_n,
                             optimize_rule=optimize_rule,
                             impulse_version=self.impulse_version,
                             evaluator_params=self.evaluator_params,
                             param_ranges=self.param_ranges)
        
        results = list(itertools.chain.from_iterable(res))
        results = pd.concat(results, axis=0) if results else pd.DataFrame()
        return results
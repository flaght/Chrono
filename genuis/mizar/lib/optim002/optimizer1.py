import importlib, pdb, os
from typing import List, Dict, Any, Tuple
import optuna
import pandas as pd
import numpy as np
from ultron.ump.core.process import add_process_env_sig, EnvProcess
from ultron.kdutils.parallel import delayed, Parallel
from kdutils.process import split_k, run_process, create_parellel
from typing import Dict, List, Any, Tuple
from lib.cux001 import FactorEvaluate1


class ImpulseCalculator(object):
    def __init__(self, impulse_version: str = 'i017', factor_names: List[str] = None):
        """
        初始化Impulse计算器

        Args:
            impulse_version: impulse版本，如'i017'
            factor_names: 要加载的因子名称列表，None表示加载所有
        """
        self.impulse_version = impulse_version

        # 动态导入impulse模块
        try:
            self.impulse_module = importlib.import_module(f'lumina.impulse.{impulse_version}')
        except ImportError as e:
            raise ImportError(f"Cannot import lumina.impulse.{impulse_version}: {e}")

        # 加载因子类
        self.factor_classes = {}
        available_factors = getattr(self.impulse_module, '__all__', None)

        if available_factors is None:
            # 如果没有__all__，尝试获取所有Impulse开头的类
            available_factors = [name for name in dir(self.impulse_module)
                               if name.startswith('Impulse') and name != 'ImpulseBase']

        if factor_names is None:
            # 加载所有因子
            factor_names = available_factors
        else:
            # 验证指定的因子名称是否存在
            available_set = set(available_factors)
            invalid_factors = [name for name in factor_names if name not in available_set]
            if invalid_factors:
                print(f"Warning: Following factors not found in {impulse_version}: {invalid_factors}")
            factor_names = [name for name in factor_names if name in available_set]

        # 导入因子类
        for factor_name in factor_names:
            if hasattr(self.impulse_module, factor_name):
                factor_class = getattr(self.impulse_module, factor_name)
                # 验证是否是ImpulseBase的子类
                try:
                    from lumina.impulse.base import ImpulseBase
                    if issubclass(factor_class, ImpulseBase):
                        self.factor_classes[factor_name] = factor_class
                except ImportError:
                    # 如果无法导入ImpulseBase，假设所有类都是有效的
                    self.factor_classes[factor_name] = factor_class

        print(f"Loaded {len(self.factor_classes)} factors from {impulse_version}:")
        pdb.set_trace()
        for name in sorted(self.factor_classes.keys()):
            param_format = self.detect_parameter_format(self.factor_classes[name])
            print(f"  - {name}: {param_format['format']} ({len(param_format['param_names'])} params)")

    def detect_parameter_format(self, factor_class) -> Dict[str, Any]:
        # 创建临时实例来检测参数格式
        temp_instance = factor_class()
        # 获取参数keys的名称 (通常以_keys结尾)
        keys_attrs = [attr for attr in dir(temp_instance) if attr.endswith('_keys')]

        if not keys_attrs:
            # 默认使用3参数格式
            return {
                'format': 'default_keys1',
                'param_names': ['window', 'weriod', 'ewm'],
                'param_types': ['int', 'int', 'categorical'],
                'default_keys': [(5, 10, 1), (10, 15, 1), (5, 10, 0), (10, 15, 0)]
            }

        keys_attr = keys_attrs[0]  # 通常只有一个keys属性
        keys_value = getattr(temp_instance, keys_attr)
        # 获取一个样本参数来判断格式
        if hasattr(keys_value, '__iter__') and len(keys_value) > 0:
            sample_params = list(keys_value)[0]
        else:
            sample_params = (5, 10, 1)  # 默认3参数

        param_length = len(sample_params) if isinstance(sample_params, (tuple, list)) else 3

        # 参数格式映射 - 根据参数长度确定格式
        format_mapping = {
            1: {
                'format': 'default_key0',
                'param_names': [],
                'param_types': [],
                'default_keys': [()]
            },
            2: {
                'format': 'default_keys0',
                'param_names': ['window', 'ewm'],
                'param_types': ['int', 'categorical'],
                'default_keys': [(5, 1), (10, 1), (5, 0), (10, 0)]
            },
            3: {
                'format': 'default_keys1',
                'param_names': ['window', 'weriod', 'ewm'],
                'param_types': ['int', 'int', 'categorical'],
                'default_keys': [(5, 10, 1), (10, 15, 1), (5, 10, 0), (10, 15, 0)]
            },
            4: {
                'format': 'default_keys2',
                'param_names': ['window', 'fast', 'slow', 'ewm'],
                'param_types': ['int', 'int', 'int', 'categorical'],
                'default_keys': [(5, 5, 10, 1), (10, 10, 15, 1), (5, 5, 10, 0), (10, 10, 15, 0)]
            },
            5: {
                'format': 'default_keys3',
                'param_names': ['window', 'fast', 'slow', 'weriod', 'ewm'],
                'param_types': ['int', 'int', 'int', 'int', 'categorical'],
                'default_keys': [(5, 5, 10, 10, 1), (10, 5, 10, 15, 1), (5, 5, 10, 10, 0), (10, 5, 10, 15, 0)]
            },
            6: {
                'format': 'default_keys8',
                'param_names': ['window', 'fast', 'medium', 'slow', 'weriod', 'ewm'],
                'param_types': ['int', 'int', 'int', 'int', 'int', 'categorical'],
                'default_keys': [(5, 5, 10, 15, 10, 1), (10, 5, 10, 15, 15, 1), (5, 5, 10, 15, 10, 0), (10, 5, 10, 15, 15, 0)]
            }
        }

        detected_format = format_mapping.get(param_length, format_mapping[3])  # 默认3参数格式

        # 使用实际的keys值
        if hasattr(keys_value, '__iter__'):
            detected_format['default_keys'] = list(keys_value)
        else:
            # 如果keys_value不是可迭代的，使用默认值
            pass

        return detected_format

    def calculate_factor_with_params(self, factor_class, params: Tuple,
                                   market_data: pd.DataFrame) -> pd.Series:
        """
        计算指定参数的因子值

        Args:
            factor_class: 因子类
            params: 参数元组 (根据因子参数格式确定)
            market_data: 市场数据 DataFrame，MultiIndex (trade_time, code)

        Returns:
            因子值Series
        """
        try:
            # 创建因子实例
            factor_instance = factor_class()

            # 根据参数设置自定义keys
            param_format = self.detect_parameter_format(factor_class)
            custom_keys = [params] if len(params) > 0 else [()]

            # 设置自定义参数
            factor_instance = factor_class(keys=custom_keys)

            # 计算因子
            result_dict = factor_instance.calc_impulse(market_data)

            # 返回第一个结果
            if result_dict and len(result_dict) > 0:
                factor_name = list(result_dict.keys())[0]
                factor_values = result_dict[factor_name]

                # 确保返回Series格式
                if isinstance(factor_values, pd.DataFrame):
                    # 如果是DataFrame，取第一个列或stack
                    if factor_values.shape[1] == 1:
                        return factor_values.iloc[:, 0]
                    else:
                        return factor_values.stack()
                elif isinstance(factor_values, pd.Series):
                    return factor_values
                else:
                    return pd.Series(dtype=float)
            else:
                return pd.Series(dtype=float)

        except Exception as e:
            print(f"Error calculating factor {factor_class.__name__} with params {params}: {e}")
            return pd.Series(dtype=float)

class ImpulseParameterOptimizer:
    """
    基于Optuna的Impulse参数优化器

    能够根据因子参数格式动态生成搜索空间，进行多目标优化。
    """
    def __init__(self, impulse_calculator: ImpulseCalculator, evaluator_template: Dict, param_ranges: Dict[str, Dict] = None):
        """
        初始化参数优化器

        Args:
            impulse_calculator: Impulse计算器实例
            evaluator_template: FactorEvaluate1的配置模板
            param_ranges: 自定义参数搜索空间范围，为None时使用默认值
        """
        self.calculator = impulse_calculator
        self.evaluator_template = evaluator_template

        # 默认参数搜索空间范围
        default_param_ranges = {
            'window': {'min': 3, 'max': 100, 'step': 1},
            'weriod': {'min': 5, 'max': 200, 'step': 5},
            'fast': {'min': 3, 'max': 50, 'step': 1},
            'slow': {'min': 5, 'max': 100, 'step': 5},
            'medium': {'min': 5, 'max': 80, 'step': 5},
            'ewm': {'choices': [0, 1]}  # 0: rolling, 1: ewm
        }

        # 使用自定义参数范围或默认值
        self.param_ranges = param_ranges if param_ranges is not None else default_param_ranges

    def get_dynamic_param_ranges(self, factor_class) -> Dict[str, Dict]:
        """
        根据因子参数格式动态生成搜索空间

        Args:
            factor_class: 因子类

        Returns:
            参数搜索空间字典
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
                  returns_data: pd.DataFrame, period: int) -> Tuple[float]:
        """
        Optuna目标函数 - 多目标优化

        Args:
            trial: Optuna试验对象
            factor_class: 要优化的因子类
            market_data: 市场数据
            returns_data: 收益率数据
            period: 预测周期

        Returns:
            目标函数值元组 (ic_mean, sharpe2, calmar)
        """
        try:
            # 动态获取参数范围
            param_ranges = self.get_dynamic_param_ranges(factor_class)
            param_format = self.calculator.detect_parameter_format(factor_class)
            
            # 根据参数格式动态采样
            sampled_params = {}
            for param_name in param_format['param_names']:
                if param_name in param_ranges:
                    range_config = param_ranges[param_name]
                    if 'choices' in range_config:
                        sampled_params[param_name] = trial.suggest_categorical(
                            param_name, range_config['choices'])
                    else:
                        sampled_params[param_name] = trial.suggest_int(
                            param_name,
                            range_config.get('min', 3),
                            range_config.get('max', 50),
                            range_config.get('step', 1))

            # 转换为参数元组（按照因子期望的顺序）
            params = tuple(sampled_params[param_name] for param_name in param_format['param_names'])
            # 计算因子
            factor_values = self.calculator.calculate_factor_with_params(
                factor_class, params, market_data)
            
            if factor_values.empty or factor_values.isna().all():
                return -1.0, -1.0, -1.0

            # 准备评估数据
            factor_data = factor_values.to_frame('transformed')
            factor_data = factor_data.reset_index()
            
            # 合并收益率数据
            ret_col = f'nxt1_ret_{period}h'
            if ret_col not in returns_data.columns:
                print(f"Warning: Return column {ret_col} not found in returns_data")
                return -1.0, -1.0, -1.0

            eval_data = factor_data.merge(
                returns_data[['trade_time', 'code', ret_col]],
                on=['trade_time', 'code'],
                how='inner'
            )

            if eval_data.empty:
                return -1.0, -1.0, -1.0

            # 创建评估器
            evaluator_config = self.evaluator_template.copy()
            evaluator_config.update({
                'factor_data': eval_data,
                'factor_name': 'transformed',
                'ret_name': ret_col,
                'roll_win': 15,
                'fee': 0.000,
                'scale_method': 'roll_zscore',
                'expression': factor_values.name,
                'name': factor_values.name,
                'resampling_win': period
            })

            evaluator = FactorEvaluate1(**evaluator_config)
            results = evaluator.run(is_check=False)

            # 提取目标指标
            ic_mean = abs(results.get('ic_mean', -1.0))
            sharpe2 = results.get('sharpe2', -1.0)
            calmar = results.get('calmar', -1.0)

            # 过滤无效结果
            if not all(np.isfinite([ic_mean, sharpe2, calmar])):
                return -1.0, -1.0, -1.0

            if ic_mean < 0.001 or sharpe2 <= 0 or calmar <= 0:
                return -1.0, -1.0, -1.0

            return ic_mean, sharpe2, calmar

        except Exception as e:
            print(f"Error in objective function for {factor_class.__name__}: {e}")
            return -1.0, -1.0, -1.0

    def optimize_factor(self, factor_class, market_data: pd.DataFrame,
                       returns_data: pd.DataFrame, period: int = 1,
                       n_trials: int = 100, timeout: int = 3600) -> Dict[str, Any]:
        """
        优化单个因子

        Args:
            factor_class: 要优化的因子类
            market_data: 市场数据
            returns_data: 收益率数据
            period: 预测周期
            n_trials: 试验次数
            timeout: 超时时间(秒)

        Returns:
            优化结果字典
        """
        print(f"Optimizing factor: {factor_class.__name__}")

        # 检测参数格式
        param_format = self.calculator.detect_parameter_format(factor_class)
        print(f"  Parameter format: {param_format['format']} ({len(param_format['param_names'])} params)")
        print(f"  Parameters: {param_format['param_names']}")

        # 创建多目标优化study
        study = optuna.create_study(
            directions=['maximize', 'maximize', 'maximize'],
            study_name=f"{factor_class.__name__}_optimization",
            sampler=optuna.samplers.NSGAIISampler()  # 多目标优化采样器
        )

        # 优化
        study.optimize(
            lambda trial: self._objective(trial, factor_class, market_data,
                                        returns_data, period),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        # 提取最佳结果
        best_trials = study.best_trials

        if not best_trials:
            print(f"  No valid results found for {factor_class.__name__}")
            return {
                'factor_name': factor_class.__name__,
                'best_params': None,
                'best_values': None,
                'all_trials': [],
                'param_format': param_format['format']
            }

        # 转换为标准格式
        results = []
        for trial in best_trials[:10]:  # 取前10个最佳结果
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
            'param_format': param_format['format'],
            'n_trials_completed': len(study.trials)
        }

        print(f"  Best IC: {best_result['best_values'][0]:.4f}")
        print(f"  Best Sharpe: {best_result['best_values'][1]:.4f}")
        print(f"  Best Calmar: {best_result['best_values'][2]:.4f}")

        return best_result


@add_process_env_sig
def run_single_factor_optimization(target_column: str, market_data: pd.DataFrame,
                                  returns_data: pd.DataFrame, period: int,
                                  n_trials: int, timeout: int, impulse_version: str,
                                  evaluator_template: Dict, param_ranges: Dict[str, Dict] = None) -> pd.DataFrame:
    """
    运行单个因子优化的函数 - 使用ultron并行处理机制
    """
    batch_results = run_process(target_column=target_column,
                               callback=optimize_single_factor,
                               market_data=market_data,
                               returns_data=returns_data,
                               period=period,
                               n_trials=n_trials,
                               timeout=timeout,
                               impulse_version=impulse_version,
                               evaluator_template=evaluator_template,
                               param_ranges=param_ranges)
    return batch_results


def optimize_single_factor(column: str, market_data: pd.DataFrame,
                          returns_data: pd.DataFrame, period: int,
                          n_trials: int, timeout: int, impulse_version: str,
                          evaluator_template: Dict, param_ranges: Dict[str, Dict] = None) -> pd.DataFrame:
    """
    优化单个因子的函数 - 参照ParallelOptimizer模式

    Args:
        column: 因子名称
        market_data: 市场数据
        returns_data: 收益率数据
        period: 预测周期
        n_trials: 试验次数
        timeout: 超时时间
        impulse_version: impulse版本
        evaluator_template: 评估器配置模板

    Returns:
        优化结果DataFrame
    """
    res = []

    try:
        pdb.set_trace()
        # 创建计算器和优化器
        calculator = ImpulseCalculator(impulse_version, [column])
        factor_class = calculator.factor_classes[column]
        optimizer = ImpulseParameterOptimizer(calculator, evaluator_template, param_ranges)

        # 执行优化
        result = optimizer.optimize_factor(
            factor_class, market_data, returns_data, period, n_trials, timeout)

        if result and not result.get('error', False):
            res.append({
                'factor_name': column,
                'best_params': result['best_params'],
                'ic_mean': result['best_values'][0] if result['best_values'] else None,
                'sharpe2': result['best_values'][1] if result['best_values'] else None,
                'calmar': result['best_values'][2] if result['best_values'] else None,
                'param_format': result.get('param_format', 'unknown')
            })

    except Exception as e:
        print(f"Error optimizing {column}: {e}")

    return pd.DataFrame(res)


class MultiFactorOptimizer:
    """
    多因子并行优化器 - 修复版本

    支持并行优化多个lumina impulse因子，能够自动检测参数格式并进行优化。
    修复了pickle序列化问题，使用独立的worker函数避免传递类对象。
    """

    def __init__(self, impulse_version: str = 'i017', factor_names: List[str] = None,
                 n_jobs: int = 4, evaluator_template: Dict = None, param_ranges: Dict[str, Dict] = None):
        """
        初始化多因子并行优化器

        Args:
            impulse_version: impulse版本
            factor_names: 要优化的因子名称列表，None表示使用所有可用因子
            n_jobs: 并行进程数
            evaluator_template: 评估器配置模板
            param_ranges: 自定义参数搜索空间范围，为None时使用默认值
        """
        # 创建impulse计算器
        self.impulse_calculator = ImpulseCalculator(impulse_version, factor_names)
        self.impulse_version = impulse_version
        self.factor_names = factor_names
        self.n_jobs = min(n_jobs, os.cpu_count() or 4)  # 不超过CPU核心数

        # 默认评估器配置
        if evaluator_template is None:
            evaluator_template = {
                'roll_win': 15,
                'fee': 0.000,
                'scale_method': 'roll_zscore',
                'annualization_factor': 252,
                'resampling_win': 1
            }

        self.evaluator_template = evaluator_template

        # 存储自定义参数范围
        self.param_ranges = param_ranges

    def optimize_all_factors_parallel(self, market_data: pd.DataFrame,
                                    returns_data: pd.DataFrame,
                                    period: int = 1, n_trials: int = 100,
                                    timeout: int = 3600) -> pd.DataFrame:
        """
        并行优化所有因子

        Args:
            market_data: 市场数据 (MultiIndex: trade_time, code)
            returns_data: 收益率数据DataFrame
            period: 预测周期
            n_trials: 每个因子的试验次数
            timeout: 单因子优化超时时间(秒)

        Returns:
            优化结果DataFrame
        """
        # 获取要优化的因子列表
        if self.factor_names is None:
            # 创建临时计算器来获取所有因子名称
            temp_calculator = ImpulseCalculator(self.impulse_version)
            factor_names = list(temp_calculator.factor_classes.keys())
        else:
            factor_names = self.factor_names

        print(f"Starting parallel optimization of {len(factor_names)} factors")
        print(f"Using {self.n_jobs} parallel processes, {n_trials} trials per factor")

        # 使用ultron并行处理机制
        factor_list = list(factor_names)
        process_list = split_k(self.n_jobs, factor_list)

        res = create_parellel(process_list=process_list,
                             callback=run_single_factor_optimization,
                             market_data=market_data,
                             returns_data=returns_data,
                             period=period,
                             n_trials=n_trials,
                             timeout=timeout,
                             impulse_version=self.impulse_version,
                             evaluator_template=self.evaluator_template,
                             param_ranges=self.param_ranges)

        # 合并结果
        import itertools
        results = list(itertools.chain.from_iterable(res))
        results_df = pd.concat(results, axis=0) if results else pd.DataFrame()

        # 处理结果
        valid_results = [r for r in results if not r.get('error', False)]
        error_results = [r for r in results if r.get('error', False)]

        print(f"\nOptimization Summary:")
        print(f"  Total factors: {len(factor_names)}")
        print(f"  Successful: {len(valid_results)}")
        print(f"  Failed: {len(error_results)}")
        print(f"  Success rate: {len(valid_results)/len(factor_names)*100:.1f}%")

        # 计算综合评分并排序
        processed_results = []
        for result in valid_results:
            if (result.get('best_params') is not None and
                result.get('best_values') is not None):

                ic_score = result['best_values'][0]
                sharpe_score = result['best_values'][1]
                calmar_score = result['best_values'][2]

                # 综合评分公式 (可调整权重)
                composite_score = (ic_score * 0.4 +
                                 sharpe_score * 0.3 +
                                 calmar_score * 0.3)

                processed_results.append({
                    'factor_name': result['factor_name'],
                    'param_format': result.get('param_format', 'unknown'),
                    'best_params': result['best_params'],
                    'ic_mean': ic_score,
                    'sharpe2': sharpe_score,
                    'calmar': calmar_score,
                    'composite_score': composite_score,
                    'n_trials_completed': result.get('n_trials_completed', n_trials)
                })

        # 转换为DataFrame并排序
        results_df = pd.DataFrame(processed_results)
        if not results_df.empty:
            results_df = results_df.sort_values('composite_score', ascending=False)
            print(f"\nTop 5 factors by composite score:")
            for i, row in results_df.head(5).iterrows():
                print(f"  {i+1}. {row['factor_name']}: {row['composite_score']:.4f} "
                      f"(IC={row['ic_mean']:.4f}, Sharpe={row['sharpe2']:.4f}, Calmar={row['calmar']:.4f})")

        return results_df

    def save_results(self, results_df: pd.DataFrame, output_dir: str = './results'):
        """
        保存优化结果

        Args:
            results_df: 优化结果DataFrame
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        if results_df.empty:
            print("No results to save")
            return

        # 保存主要结果
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'parallel_optimization_results_{timestamp}.csv'
        results_path = os.path.join(output_dir, results_file)
        results_df.to_csv(results_path, index=False)

        # 保存汇总统计
        summary_file = f'parallel_optimization_summary_{timestamp}.txt'
        summary_path = os.path.join(output_dir, summary_file)

        with open(summary_path, 'w') as f:
            f.write("Lumina Impulse Parallel Optimization Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Optimization completed at: {pd.Timestamp.now()}\n")
            f.write(f"Total factors processed: {len(results_df)}\n")
            f.write(f"Parallel processes used: {self.n_jobs}\n\n")

            if not results_df.empty:
                f.write("Performance Statistics:\n")
                f.write(f"  Best IC: {results_df['ic_mean'].max():.4f}\n")
                f.write(f"  Best Sharpe: {results_df['sharpe2'].max():.4f}\n")
                f.write(f"  Best Calmar: {results_df['calmar'].max():.4f}\n")
                f.write(f"  Average composite score: {results_df['composite_score'].mean():.4f}\n\n")

                f.write("Top 5 Factors:\n")
                for i, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
                    f.write(f"  {i}. {row['factor_name']} (Score: {row['composite_score']:.4f})\n")
                    f.write(f"     IC: {row['ic_mean']:.4f}, Sharpe: {row['sharpe2']:.4f}, Calmar: {row['calmar']:.4f}\n")
                    f.write(f"     Params: {row['best_params']}\n\n")

        print(f"Results saved to: {output_dir}")
        print(f"  - {results_file}: Main results")
        print(f"  - {summary_file}: Summary statistics")

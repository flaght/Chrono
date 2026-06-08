"""
Lumina Impulse 并行优化器 - 修复版本

修复了并发寻优时的pickle序列化错误，避免传递无法序列化的类对象。

作者: [Your Name]
日期: 2024
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'temp', 'code'))

# 导入优化器组件
from impulse_optuna_optimizer import ImpulseCalculator, ImpulseParameterOptimizer


class FixedParallelImpulseOptimizer:
    """
    修复后的并行Impulse优化器

    解决了原始版本中类对象无法序列化的问题，通过在子进程中重新创建对象来避免pickle错误。
    """

    def __init__(self, impulse_version: str = 'i017', factor_names: List[str] = None,
                 n_jobs: int = 4, evaluator_template: Dict = None):
        """
        初始化并行优化器

        Args:
            impulse_version: impulse版本
            factor_names: 要优化的因子名称列表，None表示使用所有可用因子
            n_jobs: 并行进程数
            evaluator_template: 评估器配置模板
        """
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

    def optimize_single_factor_parallel(self, task_data: Tuple) -> Dict[str, Any]:
        """
        并行优化的单个因子任务

        在子进程中重新创建所有需要的对象，避免pickle序列化问题。

        Args:
            task_data: (factor_name, market_data, returns_data, period, n_trials, timeout)

        Returns:
            优化结果字典
        """
        (factor_name, market_data, returns_data,
         period, n_trials, timeout) = task_data

        try:
            # 在子进程中重新创建所有对象
            print(f"Worker process: Optimizing {factor_name}")

            # 重新创建计算器
            calculator = ImpulseCalculator(self.impulse_version, [factor_name])
            factor_class = calculator.factor_classes[factor_name]

            # 重新创建优化器
            optimizer = ImpulseParameterOptimizer(calculator, self.evaluator_template)

            # 执行优化
            result = optimizer.optimize_factor(
                factor_class, market_data, returns_data, period, n_trials, timeout)

            print(f"Worker process: Completed {factor_name}")
            return result

        except Exception as e:
            error_msg = f"Error in worker process for {factor_name}: {str(e)}"
            print(error_msg)
            return {
                'factor_name': factor_name,
                'error': True,
                'error_message': str(e),
                'best_params': None,
                'best_values': None
            }

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

        # 准备任务 - 只传递可序列化的数据和字符串标识符
        tasks = []
        for factor_name in factor_names:
            tasks.append((
                factor_name,      # 因子名称（字符串，可序列化）
                market_data,      # 市场数据（DataFrame，可序列化）
                returns_data,     # 收益率数据（DataFrame，可序列化）
                period,           # 预测周期（整数，可序列化）
                n_trials,         # 试验次数（整数，可序列化）
                timeout           # 超时时间（整数，可序列化）
            ))

        # 并行执行
        results = []
        completed_count = 0

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # 提交所有任务
            future_to_factor = {
                executor.submit(self.optimize_single_factor_parallel, task): task[0]
                for task in tasks
            }

            # 收集结果
            for future in as_completed(future_to_factor):
                factor_name = future_to_factor[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1

                    if result.get('error'):
                        print(f"✗ Failed {completed_count}/{len(tasks)}: {factor_name} - {result.get('error_message', 'Unknown error')}")
                    else:
                        print(f"✓ Completed {completed_count}/{len(tasks)}: {factor_name}")

                except Exception as e:
                    print(f"✗ Exception in {factor_name}: {e}")
                    completed_count += 1
                    results.append({
                        'factor_name': factor_name,
                        'error': True,
                        'error_message': str(e)
                    })

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


# 使用示例函数
def create_sample_data():
    """
    创建示例数据用于测试

    Returns:
        market_data: 市场数据
        returns_data: 收益率数据
    """
    try:
        from create_data import load_random_data
        columns = ['close','low','high','open','volume','value','openint','chg', 'price']
        data = load_random_data(ticker_dim=4, factors_dim=len(columns) - 1, res_name=None)
        data = data.set_index(['trade_time', 'code'])
        data.columns = columns

        # 创建收益率数据
        returns_data = data.stack().reset_index()
        returns_data['nxt1_ret_1h'] = returns_data.groupby('code')['close'].pct_change().shift(-1)

        return data.unstack(), returns_data

    except ImportError:
        print("Warning: create_data module not found, creating dummy data")
        # 创建虚拟数据用于测试
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        codes = ['A', 'B', 'C', 'D']

        # 创建MultiIndex
        index = pd.MultiIndex.from_product([dates, codes], names=['trade_time', 'code'])

        # 创建市场数据
        np.random.seed(42)
        market_data = pd.DataFrame({
            'close': np.random.uniform(100, 200, len(index)),
            'low': np.random.uniform(95, 195, len(index)),
            'high': np.random.uniform(105, 205, len(index)),
            'open': np.random.uniform(100, 200, len(index)),
            'volume': np.random.uniform(1000, 10000, len(index)),
            'value': np.random.uniform(10000, 100000, len(index)),
            'openint': np.random.uniform(100, 1000, len(index)),
            'chg': np.random.normal(0, 0.02, len(index)),
            'price': np.random.uniform(100, 200, len(index))
        }, index=index)

        # 创建收益率数据
        returns_data = market_data.reset_index()
        returns_data['nxt1_ret_1h'] = returns_data.groupby('code')['close'].pct_change().shift(-1)

        return market_data, returns_data


def main():
    """
    主函数示例 - 演示修复后的并行优化
    """
    print("Lumina Impulse Parallel Optimization - Fixed Version")
    print("=" * 60)

    # 创建数据
    print("Loading data...")
    market_data, returns_data = create_sample_data()
    print(f"Market data shape: {market_data.shape}")
    print(f"Returns data shape: {returns_data.shape}")

    # 创建修复后的并行优化器
    optimizer = FixedParallelImpulseOptimizer(
        impulse_version='i017',
        factor_names=['ImpulseKx001', 'ImpulseKx002', 'ImpulseKx005'],  # 指定要优化的因子
        n_jobs=3  # 使用3个并行进程
    )

    # 执行并行优化
    print("\nStarting parallel optimization...")
    results = optimizer.optimize_all_factors_parallel(
        market_data=market_data,
        returns_data=returns_data,
        period=1,
        n_trials=30,  # 示例中使用较少的试验次数
        timeout=300
    )

    # 显示结果
    if not results.empty:
        print(f"\nOptimization completed successfully! Found {len(results)} valid results.")

        # 保存结果
        optimizer.save_results(results, './impulse_parallel_results')
    else:
        print("No valid optimization results found.")


if __name__ == '__main__':
    main()

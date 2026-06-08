import os, pdb
import pandas as pd
import numpy as np
from lib.fo001 import FactorOptimizer
from lib.cux001 import FactorEvaluate1

# 设置资源文件路径
operators_csv = os.path.join("records", "resource", "expression_dependencies.csv")
fields_csv = os.path.join("records", "resource", "level2_fields_dependencies.csv")

# 读取算子依赖关系
operators_pd = pd.read_csv(operators_csv).rename(
    columns={
        'Category': 'category',
        'Expression': 'expression',
        'Name': 'name',
        'Description': 'description',
        'Operator': 'operator_name',
    })

# 读取字段依赖关系
fields_pd = pd.read_csv(fields_csv).rename(
    columns={
        'types': 'field_type',
        'Field': 'field_name',
        'Formula': 'formula',
        'Description': 'description',
        'Dependencies': 'dependencies'
    })


def generate_test_data(n_samples=1000, n_stocks=100):
    """
    生成测试数据用于因子评估
    
    Args:
        n_samples: 时间序列长度
        n_stocks: 股票数量
        
    Returns:
        DataFrame: 包含价格、收益率等数据的测试数据
    """
    np.random.seed(42)

    # 生成时间序列
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    data_list = []
    for stock_id in range(n_stocks):
        # 生成价格数据（随机游走）
        price_base = 100 + np.random.randn() * 10
        returns = np.random.randn(n_samples) * 0.02  # 2%日波动率
        prices = [price_base]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # 生成成交量数据
        volume = np.random.lognormal(10, 1, n_samples)

        # 计算收益率
        forward_returns = np.roll(returns, -1)  # 前向收益率
        forward_returns[-1] = 0  # 最后一天设为0

        stock_data = pd.DataFrame({
            'trade_time':
            dates,
            'stock_id':
            f'stock_{stock_id:03d}',
            'open':
            prices,
            'high': [p * (1 + abs(np.random.randn()) * 0.01) for p in prices],
            'low': [p * (1 - abs(np.random.randn()) * 0.01) for p in prices],
            'close':
            prices,
            'volume':
            volume,
            'ret':
            forward_returns
        })

        data_list.append(stock_data)

    return pd.concat(data_list, ignore_index=True)


class CuxEvaluator:
    """基于cux001.py的因子评估器"""

    def __init__(self, test_data=None, factor_name='factor', ret_name='ret'):
        """
        初始化评估器
        
        Args:
            test_data: 测试数据，如果为None则自动生成
            factor_name: 因子列名
            ret_name: 收益率列名
        """
        if test_data is None:
            self.test_data = generate_test_data()
        else:
            self.test_data = test_data.copy()

        self.factor_name = factor_name
        self.ret_name = ret_name

    def evaluate_expression(self, expression: str, stock_id=None) -> float:
        """
        评估因子表达式
        
        Args:
            expression: 因子表达式字符串
            stock_id: 指定股票ID，如果为None则使用所有股票的平均表现
            
        Returns:
            float: 综合评分
        """
        try:
            # 这里应该根据表达式计算因子值
            # 由于表达式解析比较复杂，这里先用模拟数据
            pdb.set_trace()
            factor_values = self._simulate_factor_values(expression)

            # 选择数据
            if stock_id is not None:
                eval_data = self.test_data[self.test_data['stock_id'] ==
                                           stock_id].copy()
            else:
                # 使用所有股票的平均表现
                eval_data = self.test_data.groupby('trade_time').agg({
                    'close':
                    'mean',
                    'volume':
                    'mean',
                    'ret':
                    'mean'
                }).reset_index()
                eval_data['stock_id'] = 'average'

            # 添加因子值
            eval_data[self.factor_name] = factor_values[:len(eval_data)]

            # 使用cux001进行评估
            evaluator = FactorEvaluate1(factor_data=eval_data,
                                        factor_name=self.factor_name,
                                        ret_name=self.ret_name,
                                        roll_win=min(252,
                                                     len(eval_data) // 2),
                                        fee=0.0003,
                                        scale_method='roll_min_max',
                                        expression=expression)

            stats = evaluator.run()

            # 综合评分：IC均值 + 夏普比率 + 胜率 - 最大回撤
            score = (
                0.4 * stats.get('ic_mean', 0) +
                0.3 * stats.get('sharpe1', 0) * 0.1 +  # 夏普比率权重较小
                0.2 * stats.get('win_rate', 0) +
                0.1 * max(0, stats.get('max_dd', 0))  # 回撤惩罚
            )

            return max(0, min(1, score))  # 限制在[0,1]范围内

        except Exception as e:
            print(f"评估表达式失败: {e}")
            return 0.0

    def _simulate_factor_values(self, expression: str) -> np.ndarray:
        """
        模拟因子值计算（实际应用中应该根据表达式计算真实因子值）
        """
        # 基于表达式复杂度生成不同的因子值
        complexity = len(expression.split('('))  # 简单的复杂度度量

        # 生成具有不同特征的因子值
        n_samples = len(self.test_data)

        if 'MCORR' in expression:
            # 相关性类因子：与收益率有一定相关性
            base_factor = np.random.randn(n_samples) * 0.1
            # 添加一些与收益率的正相关性
            if hasattr(self, '_last_returns'):
                base_factor += self._last_returns * 0.3
        elif 'MA' in expression or 'AVG' in expression:
            # 移动平均类因子：相对平滑
            base_factor = np.random.randn(n_samples) * 0.05
            # 添加平滑性
            for i in range(1, len(base_factor)):
                base_factor[i] = 0.7 * base_factor[i -
                                                   1] + 0.3 * base_factor[i]
        else:
            # 其他因子：随机游走
            base_factor = np.random.randn(n_samples) * 0.08

        # 根据复杂度调整因子质量
        quality_multiplier = min(1.0, 0.5 + complexity * 0.1)
        base_factor *= quality_multiplier

        return base_factor


def rpv_objective(expression: str) -> float:
    """RPV因子目标函数（使用cux001评估）"""
    evaluator = CuxEvaluator()
    return evaluator.evaluate_expression(expression)


def simple_ma_objective(expression: str) -> float:
    """简单移动平均因子目标函数"""
    evaluator = CuxEvaluator()
    return evaluator.evaluate_expression(expression)


def momentum_objective(expression: str) -> float:
    """动量因子目标函数"""
    evaluator = CuxEvaluator()
    return evaluator.evaluate_expression(expression)


# 测试表达式
test_expressions = {
    'rpv':
    "SUBBED(MCORR(20, 'close', 'volume'), AVG(MCORR(20, 'close', 'volume')))",
    'simple_ma': "MA(20, 'close')",
    'momentum': "DELTA(5, 'close')",
    'volume_ma': "MA(10, 'volume')",
    'price_volume': "DIV(MA(5, 'close'), MA(5, 'volume'))"
}


def run_optimization_test():
    """运行优化测试"""
    print("=" * 80)
    print("因子优化测试开始")
    print("=" * 80)

    optimizer = FactorOptimizer(operators_pd=operators_pd, fields_pd=fields_pd)

    results = {}

    for name, expression in test_expressions.items():
        print(f"\n测试 {name} 因子优化...")
        print(f"原始表达式: {expression}")

        # 选择对应的目标函数
        if name == 'rpv':
            objective_func = rpv_objective
        elif name in ['simple_ma', 'volume_ma', 'price_volume']:
            objective_func = simple_ma_objective
        else:
            objective_func = momentum_objective

        # 评估原始表达式
        original_score = objective_func(expression)
        print(f"原始分数: {original_score:.4f}")

        # 运行优化
        result = optimizer.optimize_expression(
            expression=expression,
            objective_function=objective_func,
            n_trials=20,  # 减少试验次数以加快测试
            optimize_parameters=True,
            optimize_operators=True,
            study_name=f"test_{name}_optimization")

        print(f"优化后分数: {result['best_score']:.4f}")
        print(f"优化后表达式: {result['best_expression']}")
        print(f"最佳参数: {result['best_params']}")
        print(f"试验次数: {result['n_trials']}")

        results[name] = result

    return results


if __name__ == "__main__":
    # 运行测试
    results = run_optimization_test()

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

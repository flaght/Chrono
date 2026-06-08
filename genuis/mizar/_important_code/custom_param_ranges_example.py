#!/usr/bin/env python3
"""
自定义参数范围使用示例

展示如何在Lumina Impulse优化器中设置自定义参数搜索空间
"""

# 注意：实际使用时需要取消注释下面的导入
# from lib.optim002.optimizer import MultiFactorOptimizer

def example_custom_param_ranges():
    """
    自定义参数范围示例
    """

    # 示例1: 自定义窗口参数范围
    custom_ranges_narrow = {
        'window': {'min': 5, 'max': 30, 'step': 1},  # 较窄的窗口范围
        'weriod': {'min': 10, 'max': 50, 'step': 5}, # 较窄的周期范围
        'fast': {'min': 5, 'max': 20, 'step': 1},    # 较窄的快线范围
        'slow': {'min': 10, 'max': 40, 'step': 2},   # 较窄的慢线范围
        'ewm': {'choices': [0, 1]}                   # 保持默认
    }

    # 示例2: 扩展参数范围
    custom_ranges_wide = {
        'window': {'min': 2, 'max': 200, 'step': 1},    # 更宽的窗口范围
        'weriod': {'min': 3, 'max': 500, 'step': 5},    # 更宽的周期范围
        'fast': {'min': 2, 'max': 100, 'step': 1},      # 更宽的快线范围
        'slow': {'min': 3, 'max': 200, 'step': 1},      # 更宽的慢线范围
        'ewm': {'choices': [0, 1]},                     # 保持默认
        'medium': {'min': 3, 'max': 150, 'step': 3}     # 添加medium参数
    }

    # 示例3: 针对特定因子的优化参数
    custom_ranges_specific = {
        'window': {'min': 10, 'max': 60, 'step': 2},    # 针对短期因子
        'weriod': {'min': 20, 'max': 120, 'step': 10},  # 针对中期因子
        'ewm': {'choices': [1]}                          # 只使用指数加权
    }

    print("自定义参数范围配置示例")
    print("=" * 50)

    print("示例1: 窄参数范围（适用于短期交易）")
    print(f"  窗口范围: {custom_ranges_narrow['window']}")
    print(f"  周期范围: {custom_ranges_narrow['weriod']}")
    print(f"  快线范围: {custom_ranges_narrow['fast']}")
    print(f"  慢线范围: {custom_ranges_narrow['slow']}")

    print("\n示例2: 宽参数范围（适用于长期交易）")
    print(f"  窗口范围: {custom_ranges_wide['window']}")
    print(f"  周期范围: {custom_ranges_wide['weriod']}")
    print(f"  新增medium参数: {custom_ranges_wide['medium']}")

    print("\n示例3: 特定策略参数范围")
    print(f"  窗口范围: {custom_ranges_specific['window']}")
    print(f"  周期范围: {custom_ranges_specific['weriod']}")
    print(f"  EWM选择: {custom_ranges_specific['ewm']}")

    # 显示如何创建优化器（注释形式）
    print("\n创建优化器的示例代码:")
    print("# optimizer = MultiFactorOptimizer(")
    print("#     impulse_version='i017',")
    print("#     factor_names=['ImpulseKx001'],")
    print("#     param_ranges=custom_ranges_narrow")
    print("# )")

    return custom_ranges_narrow, custom_ranges_wide, custom_ranges_specific

def example_parameter_range_formats():
    """
    参数范围格式说明
    """
    print("\n参数范围格式说明")
    print("=" * 30)

    # 连续整数参数
    int_param = {
        'window': {'min': 5, 'max': 100, 'step': 5}
        # min: 最小值, max: 最大值, step: 步长
    }
    print("连续整数参数格式:")
    print(f"  {int_param}")

    # 离散选择参数
    choice_param = {
        'ewm': {'choices': [0, 1]}
        # choices: 可选值列表
    }
    print("\n离散选择参数格式:")
    print(f"  {choice_param}")

    # 浮点数参数
    float_param = {
        'alpha': {'min': 0.1, 'max': 0.9, 'step': 0.1}
        # 对于浮点数，使用相同的格式
    }
    print("\n浮点数参数格式:")
    print(f"  {float_param}")

def example_usage_in_optimization():
    """
    在优化中使用自定义参数范围
    """
    print("\n在优化中使用自定义参数范围")
    print("=" * 35)

    # 自定义参数范围 - 针对高频交易优化
    hft_param_ranges = {
        'window': {'min': 3, 'max': 15, 'step': 1},     # 很短的窗口
        'weriod': {'min': 5, 'max': 30, 'step': 1},     # 很短的周期
        'fast': {'min': 2, 'max': 10, 'step': 1},       # 快线参数
        'slow': {'min': 5, 'max': 20, 'step': 1},       # 慢线参数
        'ewm': {'choices': [1]}                          # 只使用指数加权
    }

    code_example = '''
# 使用自定义参数范围进行优化
from lib.optim002.optimizer import MultiFactorOptimizer

# 创建优化器，指定自定义参数范围
optimizer = MultiFactorOptimizer(
    impulse_version='i017',
    factor_names=['ImpulseKx001', 'ImpulseKx005'],
    param_ranges={
        'window': {'min': 3, 'max': 15, 'step': 1},    # 高频窗口
        'weriod': {'min': 5, 'max': 30, 'step': 1},    # 高频周期
        'ewm': {'choices': [1]}                         # 指数加权
    }
)

# 执行优化（参数范围会自动应用）
results = optimizer.optimize_all_factors_parallel(
    market_data=market_data,
    returns_data=returns_data,
    period=1,
    n_trials=50
)
'''

    print("代码示例:")
    print(code_example)

    print("参数范围解释:")
    for param, config in hft_param_ranges.items():
        if 'choices' in config:
            print(f"  {param}: 选择 {config['choices']}")
        else:
            print(f"  {param}: {config['min']} 到 {config['max']} (步长 {config['step']})")

if __name__ == '__main__':
    # 运行示例
    example_custom_param_ranges()
    example_parameter_range_formats()
    example_usage_in_optimization()

    print("\n" + "=" * 60)
    print("✅ 自定义参数范围功能现已可用！")
    print("您可以在创建 MultiFactorOptimizer 时通过 param_ranges 参数")
    print("指定自定义的参数搜索空间，以适应不同的交易策略需求。")
    print("=" * 60)

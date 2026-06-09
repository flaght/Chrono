#!/usr/bin/env python3
"""
测试worker函数的基本结构
"""

import sys
import os

def test_worker_function_structure():
    """测试worker函数的结构是否正确"""
    try:
        # 读取文件内容
        with open('lib/optim002/optimizer.py', 'r') as f:
            content = f.read()

        # 检查必要的元素
        checks = [
            ('def optimize_single_factor_worker', 'Worker function defined'),
            ('sys.path.insert', 'Path setup in worker'),
            ('ImpulseCalculator', 'ImpulseCalculator usage'),
            ('ImpulseParameterOptimizer', 'Optimizer usage'),
            ('return result', 'Function returns result'),
        ]

        print("Checking worker function structure:")
        all_passed = True

        for check_text, description in checks:
            if check_text in content:
                print(f"✓ {description}")
            else:
                print(f"✗ {description} - MISSING")
                all_passed = False

        # 检查函数是否在模块级别（没有缩进）
        worker_def_line = None
        for i, line in enumerate(content.split('\n')):
            if 'def optimize_single_factor_worker' in line:
                worker_def_line = line
                break

        if worker_def_line:
            if line.startswith('def '):  # 没有缩进
                print("✓ Function correctly defined at module level")
            else:
                print("✗ Function incorrectly indented")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"Error checking structure: {e}")
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("Testing Worker Function Structure")
    print("=" * 50)

    success = test_worker_function_structure()

    print("=" * 50)
    if success:
        print("✅ Worker function structure is correct!")
        print("The parallel optimization should now work.")
    else:
        print("❌ Worker function has structural issues.")
    print("=" * 50)

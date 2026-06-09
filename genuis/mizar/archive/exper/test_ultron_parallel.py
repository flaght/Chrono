#!/usr/bin/env python3
"""
测试ultron并行处理机制
"""

import sys
import os

def test_ultron_imports():
    """测试ultron相关导入是否可用"""
    try:
        from ultron.ump.core.process import add_process_env_sig, EnvProcess
        from ultron.kdutils.parallel import delayed, Parallel
        from kdutils.process import split_k, run_process, create_parellel
        print("✓ All ultron parallel imports successful")
        return True
    except ImportError as e:
        print(f"✗ Ultron import failed: {e}")
        return False

def test_split_k():
    """测试split_k函数"""
    try:
        from kdutils.process import split_k
        factor_list = ['A', 'B', 'C', 'D', 'E']
        process_list = split_k(3, factor_list)
        print(f"✓ split_k test passed: {process_list}")
        return True
    except Exception as e:
        print(f"✗ split_k test failed: {e}")
        return False

def test_optimizer_structure():
    """测试优化器结构"""
    try:
        # 检查优化器文件结构
        with open('lib/optim002/optimizer.py', 'r') as f:
            content = f.read()

        checks = [
            ('@add_process_env_sig', 'Decorator present'),
            ('run_single_factor_optimization', 'Runner function present'),
            ('optimize_single_factor', 'Worker function present'),
            ('create_parellel', 'Parallel execution present'),
            ('split_k', 'Task splitting present'),
        ]

        all_passed = True
        for check_text, description in checks:
            if check_text in content:
                print(f"✓ {description}")
            else:
                print(f"✗ {description} - MISSING")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"✗ Structure test failed: {e}")
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("Testing Ultron Parallel Processing")
    print("=" * 50)

    tests = [
        ("Ultron Imports", test_ultron_imports),
        ("Split K Function", test_split_k),
        ("Optimizer Structure", test_optimizer_structure),
    ]

    passed = 0
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        if test_func():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")

    if passed == len(tests):
        print("✅ All tests passed! Ultron parallel processing is ready.")
    else:
        print("❌ Some tests failed. Check the output above.")

    print("=" * 50)

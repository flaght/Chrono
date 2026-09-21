"""Cython 扩展构建脚本 (编译 .pyx 为 .so / .pyd)。

运行方式：
    python setup.py build_ext --inplace
"""

import os
import sys
from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# 当前目录
CURRENT_DIR = Path(__file__).resolve().parent

# 探测 bomber / nautilus_trader 头文件与 pxd 根路径
PROJECT_ROOT = CURRENT_DIR.parents[2]  # /workspace/worker/pj/Chrono/nautilus_trader_demo
BOMBER_ROOT = PROJECT_ROOT / "code" / "bomber"

include_dirs = [
    str(CURRENT_DIR),
    np.get_include(),
]

# 如果本地存在源码仓库，将 bomber 加入 pxd include 路径
if (BOMBER_ROOT / "bomber").exists():
    include_dirs.append(str(BOMBER_ROOT))

# Cython 编译器指令
compiler_directives = {
    "language_level": "3",
    "boundscheck": False,
    "wraparound": False,
    "initializedcheck": False,
    "cdivision": True,
    "embedsignature": True,
}

extensions = [
    Extension(
        name="custom_bar",
        sources=[str(CURRENT_DIR / "custom_bar.pyx")],
        include_dirs=include_dirs,
    ),
    Extension(
        name="fast_factory",
        sources=[str(CURRENT_DIR / "fast_factory.pyx")],
        include_dirs=include_dirs,
    ),
]

setup(
    name="market_cython_core",
    ext_modules=cythonize(
        extensions,
        include_path=include_dirs,
        compiler_directives=compiler_directives,
    ),
)

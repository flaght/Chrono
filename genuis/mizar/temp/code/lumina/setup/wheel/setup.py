# -*- coding: utf-8 -*-
import os
import sys
import platform
import multiprocessing
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py as _build_py
from Cython.Build import cythonize

# --- Package Info ---
PACKAGE_NAME_TO_SCAN = "lumina"

# --- Cythonization Configuration ---
# 定义不希望被Cython编译的文件列表 (相对于项目根目录)
EXCLUDE_FROM_CYTHON = {
    os.path.normpath(os.path.join(PACKAGE_NAME_TO_SCAN, 'genetic', 'strategy', 'method', 'atr.py')),
    os.path.normpath(os.path.join(PACKAGE_NAME_TO_SCAN, 'genetic', 'strategy', 'method', 'trailing.py')),
    os.path.normpath(os.path.join(PACKAGE_NAME_TO_SCAN, 'genetic', 'fusion', 'orders', 'orders_nb.py'))
}

# --- Helper function to find .py files for Cythonization ---
def get_extensions(package_name):
    extensions = []
    # 如果启用了代码覆盖率行跟踪
    enable_line_trace = "--line_trace" in sys.argv
    if enable_line_trace:
        print("Build with line trace enabled ...")
        sys.argv.remove("--line_trace")
        define_macros = [('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')]
    else:
        define_macros = []

    for root, _, files in os.walk(package_name):
        for file in files:
            if not file.endswith(".py"):
                continue

            py_path = os.path.join(root, file)
            # 检查是否在排除列表中
            if os.path.normpath(py_path) in EXCLUDE_FROM_CYTHON:
                print(f"[get_extensions] SKIPPING (in exclude list): {py_path}")
                continue

            # 忽略 __init__.py 文件，除非你确实想编译它们
            if file == '__init__.py':
                continue

            # 将文件路径转换为模块导入路径 (e.g., lumina/core/utils.py -> lumina.core.utils)
            module_path = os.path.splitext(py_path)[0].replace(os.sep, '.')

            print(f"[get_extensions] ADDING for compilation: {py_path} -> {module_path}")
            extensions.append(
                Extension(
                    name=module_path,
                    sources=[py_path],
                    define_macros=define_macros
                )
            )
    return extensions

# --- Custom build_py to skip source files of compiled modules ---
class build_py_skip_cythonized(_build_py):
    """
    A custom build_py command that filters out .py files that have been
    compiled into .so extension modules.
    """
    def find_package_modules(self, package, package_dir):
        if self.distribution.ext_modules:
            compiled_modules = {ext.name for ext in self.distribution.ext_modules}
        else:
            compiled_modules = set()

        modules = super().find_package_modules(package, package_dir)

        # The core logic: filter out .py files that match a compiled module name
        filtered_modules = []
        for (pkg, module, file) in modules:
            module_name = f"{pkg}.{module}"
            if module_name not in compiled_modules:
                filtered_modules.append((pkg, module, file))
            else:
                print(f"[build_py] SKIPPING source file for compiled module: {file}")

        return filtered_modules

# --- Setup configuration ---
n_threads = 0
if platform.system() != "Windows":
    try:
        n_threads = multiprocessing.cpu_count()
    except NotImplementedError:
        n_threads = 1 # Fallback

py_extensions = get_extensions(PACKAGE_NAME_TO_SCAN)
line_trace_enabled = "--line_trace" in sys.argv

setup(
    # Project metadata is now in pyproject.toml, so we don't repeat it here
    # unless necessary for specific setup.py logic.
    packages=find_packages(exclude=["tests*"]),
    ext_modules=cythonize(
        py_extensions,
        compiler_directives={
            'language_level': "3",
            'linetrace': line_trace_enabled,
        },
        nthreads=n_threads,
        # Force compilation even if .c files are up-to-date
        force=True, 
        # Create stubs for type checkers
        # annotate=True, # Uncomment to generate HTML annotation files
    ) if py_extensions else [],
    cmdclass={
        'build_py': build_py_skip_cythonized,
    },
    # This is important! It tells pip that the wheel is not "universal"
    # and contains compiled code specific to a platform.
    zip_safe=False,
    include_package_data=True,
)
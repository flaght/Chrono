"""
编译命令：
    cd /path/to/orion/lib/cms002
    python setup.py build_ext --inplace
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    name="booster",  # 不带包前缀，直接输出 booster.so 到当前目录
    sources=["booster.pyx"],
    include_dirs=[numpy.get_include()],
)

setup(
    name='booster',
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
        }),
)

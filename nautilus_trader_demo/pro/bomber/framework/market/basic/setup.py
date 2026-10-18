"""Build the Cython extensions using their published module names.

Run with the target engine's Python: python bomber/framework/market/basic/setup.py
build_ext --inplace. The working directory is normalized to the package root.
"""
import os
import sys
from pathlib import Path

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))
import bomber

# Both the local framework and the installed engine provide Cython interfaces.
include_dirs = list(dict.fromkeys([
    str(CURRENT_DIR), str(PACKAGE_ROOT), np.get_include(),
    *(str(Path(location).resolve().parent) for location in bomber.__path__),
]))
compiler_directives = {
    "language_level": "3", "boundscheck": False, "wraparound": False,
    "initializedcheck": False, "cdivision": True, "embedsignature": True,
}
extensions = [
    Extension(
        name=f"bomber.framework.market.basic.{name}",
        sources=[str(CURRENT_DIR / f"{name}.pyx")],
        include_dirs=include_dirs,
    )
    for name in ("custom_bar", "fast_factory")
]
os.chdir(PACKAGE_ROOT)
setup(
    name="bomber-framework-market-core",
    ext_modules=cythonize(extensions, include_path=include_dirs,
                          compiler_directives=compiler_directives),
)

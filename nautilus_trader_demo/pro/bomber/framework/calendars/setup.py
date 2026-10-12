"""Build the calendar extensions with the Python that will use them.

From any directory:
    python /path/to/pro/bomber/framework/calendars/setup.py build_ext --inplace
"""

import os
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup


CALENDAR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CALENDAR_DIR.parents[2]
BUILD_DIR = PROJECT_ROOT / ".build" / "calendars"


def main():
    # Full module names must match both Python imports and Cython cimports.
    extensions = [
        Extension(
            ".".join(source.relative_to(PROJECT_ROOT).with_suffix("").parts),
            [str(source)],
            include_dirs=[str(PROJECT_ROOT)],
        )
        for source in sorted(CALENDAR_DIR.rglob("*.pyx"))
    ]
    os.chdir(PROJECT_ROOT)
    setup(
        name="bomber-framework-calendars",
        version="0.1.0",
        packages=[],
        py_modules=[],
        ext_modules=cythonize(
            extensions,
            include_path=[str(PROJECT_ROOT)],
            build_dir=str(BUILD_DIR / "cython"),
            compiler_directives={"language_level": 3},
        ),
        options={"build": {"build_base": str(BUILD_DIR)}},
    )


if __name__ == "__main__":
    main()

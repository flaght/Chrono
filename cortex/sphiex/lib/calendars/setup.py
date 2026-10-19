"""Build the calendar extensions with the Python that will use them.

From any directory:
    python /path/to/calendars/setup.py build_ext --inplace

For a nested import such as lib.calendars:
    CALENDARS_PACKAGE=lib.calendars python /path/to/calendars/setup.py build_ext --inplace --force
"""

import os
import sys
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup


CALENDAR_DIR = Path(__file__).resolve().parent


def build_config():
    package_name = os.environ.get("CALENDARS_PACKAGE", CALENDAR_DIR.name)
    parts = package_name.split(".")
    if not all(part.isidentifier() for part in parts):
        raise ValueError("CALENDARS_PACKAGE must be a dotted Python package name")
    package_root = CALENDAR_DIR
    for part in reversed(parts):
        if package_root.name != part:
            raise ValueError(
                f"CALENDARS_PACKAGE={package_name!r} does not match {CALENDAR_DIR}"
            )
        package_root = package_root.parent
    # Different namespaces must not reuse C sources containing old cimport names.
    build_dir = CALENDAR_DIR / ".build" / package_name
    return package_name, package_root, build_dir


def main():
    package_name, package_root, build_dir = build_config()
    # Cython cimports bind to these full names, even with relative source imports.
    extensions = [
        Extension(
            package_name + "." + ".".join(
                source.relative_to(CALENDAR_DIR).with_suffix("").parts
            ),
            [str(source)],
            include_dirs=[str(package_root)],
        )
        for source in sorted(CALENDAR_DIR.rglob("*.pyx"))
    ]
    os.chdir(package_root)
    setup(
        name="calendars",
        version="0.1.0",
        packages=[],
        py_modules=[],
        ext_modules=cythonize(
            extensions,
            include_path=[str(package_root)],
            build_dir=str(build_dir / "cython"),
            force="--force" in sys.argv,
            compiler_directives={"language_level": 3},
        ),
        options={"build": {"build_base": str(build_dir)}},
    )


if __name__ == "__main__":
    main()

# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "order_cy",  # This is the name you'll use to import the module
        ["kdutils/orders/order_cy.pyx"],
        include_dirs=[numpy.get_include()], # Necessary for cimport numpy
        # define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")] # Optional: suppress NumPy deprecation warnings
        language="c++" # Optional: can sometimes allow for more optimizations or C++ lib usage
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3", # Use Python 3 syntax
            # 'boundscheck': False, # Uncomment for potential speedup, but be careful
            # 'wraparound': False,  # Uncomment for potential speedup
            # 'cdivision': True,    # Use C-style division for integers
        }
    ),
)
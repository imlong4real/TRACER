"""Build hook for Cython extensions.

Package metadata lives in pyproject.toml. This file exists solely to
compile the .pyx files into C extensions at install time, because the
pyproject-only setuptools backend does not yet handle ext_modules cleanly.
"""
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

extensions = cythonize(
    [
        "src/tracer/_cy_prune.pyx",
        "src/tracer/_cy_spatial.pyx",
    ],
    language_level=3,
    compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "nonecheck": False,
    },
)

for ext in extensions:
    ext.include_dirs.append(np.get_include())

setup(ext_modules=extensions)

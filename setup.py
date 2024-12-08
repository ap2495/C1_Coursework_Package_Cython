from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "dual_autodiff_x.dual",
        ["src/dual_autodiff_x/dual.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="dual_autodiff_x",
    version="0.0.1b2",
    packages=["dual_autodiff_x"],
    package_dir={"": "src"},
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"}  # ensure Python 3 semantics
    ),
    package_data={"dual_autodiff_x": ["*.so", "*.pyd"]},
    exclude_package_data={"dual_autodiff_x": ["*.pyx", "*.py"]},
    zip_safe=False,
)

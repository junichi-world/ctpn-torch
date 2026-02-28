from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

numpy_include = np.get_include()

extensions = [
    Extension(
        "bbox",
        ["bbox.pyx"],
        include_dirs=[numpy_include],
    ),
    Extension(
        "cython_nms",
        ["cython_nms.pyx"],
        include_dirs=[numpy_include],
    ),
]

setup(
    ext_modules=cythonize(extensions),
)
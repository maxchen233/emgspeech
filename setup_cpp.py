"""
Build the cpp_forward pybind11 module.

Usage:
    pip install pybind11
    python setup_cpp.py build_ext --inplace
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext = Pybind11Extension(
    "cpp_forward",
    sources=[
        "cpp_inference/src/pybind_module.cpp",
        "cpp_inference/src/ops.cpp",
        "cpp_inference/src/emg_model.cpp",
    ],
    include_dirs=["cpp_inference/src", "/usr/include/openblas"],
    cxx_std=17,
    extra_compile_args=["-O3", "-march=native"],
    libraries=["openblas"],
)

setup(
    name="cpp_forward",
    version="0.1.0",
    description="C++ EMG-to-Mel inference engine",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)

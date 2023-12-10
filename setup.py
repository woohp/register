#!/usr/bin/env python
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "register._ext",
        [
            "src/module.cpp",
        ],
        language="c++",
        extra_compile_args=["-std=c++2a"],
        # define_macros=[('VERSION_INFO', __version__)],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

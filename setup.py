#!/usr/bin/env python
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        '_ext',
        [
            'src/module.cpp',
        ],
        language='c++',
        # define_macros=[('VERSION_INFO', __version__)],
    ),
]

setup(
    name='_ext',
    # description='PDF Renderer',
    version='0.0.1',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    test_suite='tests',
    zip_safe=False,
)

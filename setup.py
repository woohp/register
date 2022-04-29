#!/usr/bin/env python
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

ext_modules = [
    Pybind11Extension(
        'register._ext',
        [
            'src/module.cpp',
        ],
        language='c++',
        extra_compile_args=['-std=c++2a', '-march=skylake'],
        # define_macros=[('VERSION_INFO', __version__)],
    ),
]

setup(
    name='register',
    description='Image Registration',
    version='0.0.1',
    packages=find_packages(exclude=('tests', )),
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    test_suite='tests',
    zip_safe=False,
)

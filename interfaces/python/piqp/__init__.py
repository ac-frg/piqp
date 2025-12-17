# This file is part of PIQP.
#
# Copyright (c) 2023 EPFL
# Copyright (c) 2022 INRIA
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

from . import instruction_set
import os


def load_main_module(globals):
    def load_module(main_module_name):
        import importlib

        try:
            main_module = importlib.import_module("." + main_module_name, __name__)
            globals.update(main_module.__dict__)
            del globals[main_module_name]
            return True
        except ModuleNotFoundError:
            return False

    all_modules = [
        ("piqp_python_avx512", instruction_set.avx512f),
        ("piqp_python_avx2", instruction_set.avx2),
    ]

    for module_name, check in all_modules:
        if check and load_module(module_name):
            return

    assert load_module("piqp_python")


def get_include():
    """Return the directory containing piqp C++ headers.

    Returns
    -------
    str
        Absolute path to the directory containing piqp headers.
    """
    return os.path.join(os.path.dirname(__file__), "include")


def get_library_dir():
    """Return the directory containing piqp shared libraries.

    Returns
    -------
    str
        Absolute path to the directory containing piqp shared libraries.
    """
    return os.path.join(os.path.dirname(__file__), "lib")


def get_cmake_dir():
    """Return the directory containing piqp CMake config files.

    Returns
    -------
    str
        Absolute path to the directory containing CMake config files.
    """
    return os.path.join(os.path.dirname(__file__), "cmake")


load_main_module(globals=globals())
del load_main_module

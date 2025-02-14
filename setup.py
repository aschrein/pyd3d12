# from skbuild import setup
from __future__ import annotations
import importlib.util
import setuptools
import skbuild
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from sys import platform
import sys
import os

# https://scikit-build.readthedocs.io/en/latest/usage.html

build_native_pass = "build" in sys.argv

cmake_process_manifest_hook = None
cmake_source_dir = "."

CONSOLE_COLOR_RED = "\033[91m"
CONSOLE_COLOR_GREEN = "\033[92m"
CONSOLE_COLOR_END = "\033[0m"

if not build_native_pass:
    # sys.argv.append("--skip-cmake")
    cmake_process_manifest_hook = lambda a: []
    cmake_source_dir = None
    print(CONSOLE_COLOR_RED, "---------------- Skipping native build ----------------", CONSOLE_COLOR_END)
else:
    print(CONSOLE_COLOR_GREEN, "---------------- Building native ----------------", CONSOLE_COLOR_END)

os.environ['TEMP'] = os.path.abspath('./build')
os.environ['TMP'] = os.path.abspath('./build')

lib_list = [
    "pybind11",
    "sciking-build",
    "sciking-build-core",
    "numpy",
    # "Pygments",
    # "PyQt5",
    # "PyQt5-sip",
    # "PyQt5-Qt5",
    # "QDarkStyle",
    # "watchdog",
    # "QScintilla",
]

setuptools.setup(
    cmake_args=["-DCMAKE_INSTALL_PREFIX=" + str(os.path.abspath('./build'))],
    cmake_install_dir=os.path.abspath('./build'),
    name="pyd3d12",
    version="0.0",
    packages=['native', 'py', 'tests'],
    cmake_source_dir=cmake_source_dir,
    cmake_process_manifest_hook=cmake_process_manifest_hook,
)
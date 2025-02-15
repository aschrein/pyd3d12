# MIT License
# 
# Copyright (c) 2025 Anton Schreiner
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os, sys
from pathlib import Path
import importlib.util
import psutil
import subprocess
import time

CONSOLE_COLOR_RED = "\033[91m"
CONSOLE_COLOR_GREEN = "\033[92m"
CONSOLE_COLOR_END = "\033[0m"

def launch_debugviewpp(exe_path=r"bin\\DebugViewpp.exe"):
    # Check if an instance is already running.
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] and proc.info['name'].lower() == "debugviewpp.exe":
            print("DebugView++ is already running.")
            return

    # Launch DebugView++ if not running.
    if os.path.exists(exe_path):
        subprocess.Popen([exe_path])
        # sleep for 1 second to allow DebugView++ to start.
        time.sleep(1)
        print("Launched DebugView++.")
    else:
        print(f"Executable not found: {exe_path}")

def quote_path(path):
    if isinstance(path, Path):
        path = "\"" + str(path) + "\""
        return path
    if isinstance(path, str):
        path = Path(path)
        path = "\"" + str(path) + "\""
    return path

def find_file_or_folder(file_name, folder=None):
    """
        Return the path of the first folder containing the file_name, including the file_name
    """
    if folder is None:
        folder = Path(__file__).parent

    file_name = Path(file_name)

    while folder != None:
        if os.path.exists(folder / file_name):
            return folder / file_name

        folder = folder.parent
    
    return None

def get_third_party_folder():
    return find_file_or_folder("third_party")

def get_bin_folder():
    return find_file_or_folder("bin")

def get_or_create_tmp_folder():
    bin_folder = get_bin_folder()
    temp_folder = bin_folder.parent / ".tmp"
    os.makedirs(temp_folder, exist_ok=True)
    return temp_folder

def find_native_module_path(name):
    """
        Try to find the native module in the build folder
    """

    build_type = os.environ.get("NATIVE_BUILD_TYPE", "Release")

    build_folder = find_file_or_folder("build")

    python_version = f"{sys.version_info.major}{sys.version_info.minor}"

    search_folder = build_folder / f"temp.win-amd64-cpython-{python_version}\\Release\\_skbuild\\native\\{build_type}"

    module_name = f"{name}.cp{python_version}-win_amd64.pyd"

    print(f"Searching for {module_name} in {search_folder}")
    if os.path.exists(search_folder / module_name):
        return search_folder / module_name
    
    return None

def find_native_module(name):
    """
        Try to find the native module in the build folder
    """

    path = find_native_module_path(name)
    folder = path.parent

    if path is not None:
        # spec = importlib.util.spec_from_file_location(name, path)
        # module = importlib.util.module_from_spec(spec)
        # spec.loader.exec_module(module)

        sys.path.insert(0, str(folder))
        os.add_dll_directory(str(folder))
        return __import__(name)
    
    raise ImportError(f"Native module {name} not found")
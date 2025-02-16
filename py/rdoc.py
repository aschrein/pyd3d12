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
from .utils import *

rdoc = find_native_module("rdoc")

rdoc_path = None

def find_rdoc():
    global rdoc_path
    if rdoc_path is not None:
        return rdoc_path
    
    try_path = "C:\\Program Files\\RenderDoc"
    if os.path.exists(try_path):
        print("RenderDoc found at: " + try_path)
        rdoc_path = Path(try_path) / "renderdoc.dll"
        return rdoc_path
    
    return None

ctx = None

def rdoc_is_valid():
    return ctx is not None and ctx.IsValid()

def rdoc_load():
    global ctx
    rdoc_path = find_rdoc()
    if rdoc_path is None:
        print("RenderDoc not found.")
        return False
    # dll_load_path = str(rdoc_path.parent)
    # print("Adding DLL directory: " + dll_load_path)
    # os.add_dll_directory(dll_load_path)

    ctx = rdoc.CreateContext(str(rdoc_path))
    assert ctx.IsValid()

    ctx.SetCaptureFilePathTemplate(str(get_or_create_tmp_folder() / "capture.rdc"))


def rdoc_start_capture():
    if not rdoc_is_valid():
        print("RenderDoc not loaded.")
        return
    assert ctx is not None
    ctx.StartCapture()
    print("Started capture.")

def rdoc_end_capture():
    if not rdoc_is_valid():
        print("RenderDoc not loaded.")
        return
    assert ctx is not None
    ctx.EndCapture()
    print("Ended capture.")

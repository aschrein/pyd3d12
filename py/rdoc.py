# MIT License
# Copyright (c) 2025 Anton Schreiner


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
        if os.environ.get("DEBUG", "0") != "0": print_red("RenderDoc not loaded.")
        return
    assert ctx is not None
    ctx.StartCapture()
    
    if os.environ.get("DEBUG", "0") != "0": print_green("Started capture.")

def rdoc_end_capture():
    if not rdoc_is_valid():
        if os.environ.get("DEBUG", "0") != "0": print_red("RenderDoc not loaded.")
        return
    assert ctx is not None
    ctx.EndCapture()
    if os.environ.get("DEBUG", "0") != "0": print_green("Ended capture.")

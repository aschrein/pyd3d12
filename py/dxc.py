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

from py.utils import *
import sys, os, subprocess

def split_cmd_args_into_list(args):
    assert isinstance(args, str)
    return args.split(" ")

class DXCContext:
    def __init__(self):
        bin_folder = get_bin_folder()
        dxc_version = os.environ.get("DXC_VERSION", "dxc_2024_07_31")
        self.dxc_path = bin_folder / dxc_version / "bin/x64/dxc.exe"
        self.tmp_path = get_or_create_tmp_folder() / "dxc"
        os.makedirs(self.tmp_path, exist_ok=True)
        pass

    def compile_to_dxil(self, source, args):
        if isinstance(source, Path):
            source = open(source, "r").read()
        if isinstance(args, str):
            args = split_cmd_args_into_list(args)
        
        _hash = hash(source)
        dst = str(self.tmp_path / f"{_hash}.dxil")

        # write source to file
        source_file = open(self.tmp_path / f"{_hash}.hlsl", "w")
        source_file.write(source)
        source_file.close()

        args = [str(self.dxc_path)] + args + [str(self.tmp_path / f"{_hash}.hlsl"), "/Fo", str(dst)]
        # print(f"args = {args}")
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # process.stdin.write(source.encode())
        process.wait()
        if process.returncode != 0:
            print(f"Error: {process.stderr.read().decode()}")
            return None
        
        bytecode = open(dst, "rb").read()
        return bytecode

        
        

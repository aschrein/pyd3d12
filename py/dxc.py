# MIT License
# Copyright (c) 2025 Anton Schreiner

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
        self.include_paths = []
        hlsl_dir = find_file_or_folder("common/hlsl")
        if hlsl_dir:
            self.include_paths.append("-I")
            self.include_paths.append(str(hlsl_dir))
        self.default_args = ["-HV", "2021", "-enable-16bit-types"]
        os.makedirs(self.tmp_path, exist_ok=True)
        pass

    def add_include_path(self, path):
        self.include_paths.append("-I")
        self.include_paths.append(str(path))
        return self

    def compile_to_dxil(self, source, args):
        if isinstance(args, str):
            args = split_cmd_args_into_list(args)

        if isinstance(source, Path):
            _hash = hash(str(source))
            dst = str(self.tmp_path / f"{_hash}.dxil")
            args = [str(self.dxc_path)] + self.include_paths + self.default_args + args + [f"{str(source)}", "/Fo", str(dst)]
            if get_debug_level(): print_purple(f"args = {args}")
            process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.wait()
            if process.returncode != 0:
                print_red(f"Error: {process.stderr.read().decode()}")
                return None
            
            bytecode = open(dst, "rb").read()
            return bytecode


        _hash   = hash(source)
        dst     = str(self.tmp_path / f"{_hash}.dxil")

        # write source to file
        source_file = open(self.tmp_path / f"{_hash}.hlsl", "w")
        source_file.write(source)
        source_file.close()

        args = [str(self.dxc_path)] + self.include_paths + self.default_args + args + [str(self.tmp_path / f"{_hash}.hlsl"), "/Fo", str(dst)]
        if get_debug_level(): print_purple(f"args = {args}")
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # process.stdin.write(source.encode())
        process.wait()
        if process.returncode != 0:
            print_red(f"Error: {process.stderr.read().decode()}")
            return None
        
        bytecode = open(dst, "rb").read()
        return bytecode

        
        

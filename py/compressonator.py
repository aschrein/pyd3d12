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
from py.utils import *
import subprocess

def find_compressonator():
    try_path = "C:\\Program Files\\Compressonator"
    if os.path.exists(try_path):
        print("Compressonator found at: " + try_path)
        return try_path

    bin_folder = get_bin_folder()

    # Try to find the comporessonator* folder in the bin folder
    for attempt in range(2):
        for entry in os.listdir(bin_folder):
            if entry.startswith("compressonator"):
                try_path = bin_folder / entry
                if os.path.exists(try_path):
                    print("Compressonator found at: " + str(try_path))
                    return try_path
        # launch scripts/fetch compressonator to get the latest version
        os.system(str(bin_folder.parent) + "\\scripts\\fetch.bat compressonator")

    return None


def find_compressonator_cli():
    compressonator_path = find_compressonator()
    if compressonator_path is None:
        print("Compressonator not found.")
        return None

    try_path = compressonator_path / "compressonatorcli.exe"
    if os.path.exists(try_path):
        print("Compressonator CLI found at: " + str(try_path))
        return try_path

    return None

def compressonator_cli_run(args):
    compressonator_cli_path = find_compressonator_cli()
    if compressonator_cli_path is None:
        print("Compressonator CLI not found.")
        return False

    args = [str(compressonator_cli_path)] + args
    print("Running: " + " ".join(args))
    subprocess.run(args)

def compress(input_file, output_file, num_mips=8, format="BC7"):
    # compressonator_cli_run(["-fd", "BC7", "-EncodeWith", "GPU", "-miplevels", str(num_mips), "-GenGPUMipMaps", input_file, output_file])
    compressonator_cli_run(["-fd", format, "-EncodeWith", "CPU", "-miplevels", str(num_mips), input_file, output_file])

if __name__ == "__main__":
    # print(find_compressonator_cli())
    import argparse

    parser = argparse.ArgumentParser(description="Compressonator CLI")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument("--output_file", type=str, help="Output file")
    parser.add_argument("--format", type=str, help="Output format", default="BC7")
    parser.add_argument("--num_mips", type=int, help="Number of mip levels", default=8)

    args = parser.parse_args()

    compress(args.input_file, args.output_file, args.num_mips, args.format)


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
from py.dds import *
from py.utils import *
from py.d3d12 import *
from py.rdoc import *

launch_debugviewpp()

rdoc_load()

debug = native.ID3D12Debug()
debug.EnableDebugLayer()
factory = native.IDXGIFactory()
adapters = factory.EnumAdapters()
device = native.CreateDevice(adapters[0], native.D3D_FEATURE_LEVEL._11_0)

asset_folder = find_file_or_folder("assets")
assert asset_folder is not None

dds_file = DDSTexture(asset_folder / "mandrill.dds")

rdoc_start_capture()
texture = make_texture_from_dds(device, dds_file)
rdoc_end_capture()
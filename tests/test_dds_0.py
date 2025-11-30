# MIT License
# Copyright (c) 2025 Anton Schreiner

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
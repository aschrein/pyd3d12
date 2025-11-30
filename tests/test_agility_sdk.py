# MIT License
# Copyright (c) 2025 Anton Schreiner

import os, sys
from py.dds import *
from py.utils import *
from py.d3d12 import *
from py.rdoc import *

launch_debugviewpp()

bin_folder                  = get_bin_folder()
agility_sdk_folder          = bin_folder / "Microsoft.Direct3D.D3D12.1.615.0\\build\\native\\bin\\x64"
agility_sdk_version         = 615
exe_rel_agility_sdk_folder  = os.path.relpath(agility_sdk_folder, sys.executable)
d3d_config                  = native.ID3D12SDKConfiguration()
d3d_config.SetSDKVersion(agility_sdk_version, exe_rel_agility_sdk_folder)

debug = native.ID3D12Debug()
debug.EnableDebugLayer()
factory = native.IDXGIFactory()
adapters = factory.EnumAdapters()
device = native.CreateDevice(adapters[0], native.D3D_FEATURE_LEVEL._11_0)

asset_folder = find_file_or_folder("assets")
assert asset_folder is not None

dds_file = DDSTexture(asset_folder / "mandrill.dds")
texture = make_texture_from_dds(device, dds_file)

print(CONSOLE_COLOR_GREEN + "SUCCESS" + CONSOLE_COLOR_END)

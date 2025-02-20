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
import ctypes
import argparse
from py.utils import *

args = argparse.ArgumentParser()
args.add_argument("--build", type=str, default="Release")
args.add_argument("--wait_for_debugger_present", action="store_true")
args.add_argument("--load_rdoc", action="store_true")
args.add_argument("--load_pix", action="store_true")
args.add_argument("--make_window", action="store_true")
args = args.parse_args()
set_build_type(args.build)

from py.dxc import *
from py.rdoc import *
from py.d3d12 import *
from py.pix import *

launch_debugviewpp()

hlsl_folder = Path(__file__).parent / "hlsl"
assert hlsl_folder.exists()

if args.load_pix:
    pix.PIXLoadLatestWinPixGpuCapturerLibrary()
    # pix.PIXBeginCapture()

native = find_native_module("native")

if args.load_rdoc and  find_rdoc() is not None:
    rdoc_load()

debug = native.ID3D12Debug()
debug.EnableDebugLayer()

factory = native.IDXGIFactory()
adapters = factory.EnumAdapters()
device = native.CreateDevice(adapters[0], native.D3D_FEATURE_LEVEL._11_0)

dxc_ctx = DXCContext()
bytecode = dxc_ctx.compile_to_dxil(source = hlsl_folder / "mlp_test.hlsl", args = "-E Main -T cs_6_5",)

signature = device.CreateRootSignature(
    Bytes = bytecode
)
assert signature is not None
Main_pso = device.CreateComputePipelineState(native.D3D12_COMPUTE_PIPELINE_STATE_DESC(
    RootSignature = signature,
    CS = native.D3D12_SHADER_BYTECODE(bytecode),
    NodeMask = 0,
    CachedPSO = None,
    Flags = native.D3D12_PIPELINE_STATE_FLAGS.NONE
))
Backwards_pso = device.CreateComputePipelineState(native.D3D12_COMPUTE_PIPELINE_STATE_DESC(
    RootSignature = signature,
    CS = native.D3D12_SHADER_BYTECODE(dxc_ctx.compile_to_dxil(source = hlsl_folder / "mlp_test.hlsl", args = "-E Backward -T cs_6_5",)),
    NodeMask = 0,
    CachedPSO = None,
    Flags = native.D3D12_PIPELINE_STATE_FLAGS.NONE
))
Inference_pso = device.CreateComputePipelineState(native.D3D12_COMPUTE_PIPELINE_STATE_DESC(
    RootSignature = signature,
    CS = native.D3D12_SHADER_BYTECODE(dxc_ctx.compile_to_dxil(source = hlsl_folder / "mlp_test.hlsl", args = "-E InferencePass -T cs_6_5",)),
    NodeMask = 0,
    CachedPSO = None,
    Flags = native.D3D12_PIPELINE_STATE_FLAGS.NONE
))
Initialize_pso = device.CreateComputePipelineState(native.D3D12_COMPUTE_PIPELINE_STATE_DESC(
    RootSignature = signature,
    CS = native.D3D12_SHADER_BYTECODE(dxc_ctx.compile_to_dxil(source = hlsl_folder / "mlp_test.hlsl", args = "-E Initialize -T cs_6_5",)),
    NodeMask = 0,
    CachedPSO = None,
    Flags = native.D3D12_PIPELINE_STATE_FLAGS.NONE
))


cbv_srv_uav_heap = CBV_SRV_UAV_DescriptorHeap(device, 1024)
cpu_handle, gpu_handle = cbv_srv_uav_heap.get_next_descriptor_handle(16)
srv_cpu_handle, srv_gpu_handle = cbv_srv_uav_heap.get_next_descriptor_handle(16)
asset_folder = find_file_or_folder("assets")
assert asset_folder is not None

dds_file            = DDSTexture(asset_folder / "mandrill.dds")
width               = dds_file.header.width
height              = dds_file.header.height
target_texture      = make_texture_from_dds(device, dds_file)
uav_texture         = make_uav_texture_2d(device, width=dds_file.header.width, height=dds_file.header.height, format=native.DXGI_FORMAT.R16G16B16A16_FLOAT)

pitch               = 4 * 4 * width
storage_buffer_size = pitch * height
params_storage_size = 16 * (1 << 20); # 16mb should be enough for now

params_buffer   = make_uav_buffer(device, params_storage_size)
grads_buffer    = make_uav_buffer(device, params_storage_size)

device.CreateUnorderedAccessView(
        Resource=uav_texture,
        Desc = native.D3D12_UNORDERED_ACCESS_VIEW_DESC(
            Format = uav_texture.GetDesc().Format,
            Texture2D = native.D3D12_TEX2D_UAV(
                MipSlice = 0,
                PlaneSlice = 0
            )
        ),
        DestDescriptor = native.D3D12_CPU_DESCRIPTOR_HANDLE(cpu_handle.ptr + 0),
        CounterResource = None
    )

device.CreateUnorderedAccessView(
        Resource=params_buffer,
        Desc = native.D3D12_UNORDERED_ACCESS_VIEW_DESC(
            Format = native.DXGI_FORMAT.UNKNOWN,
            Buffer = native.D3D12_BUFFER_UAV(
                FirstElement = 0,
                NumElements = params_storage_size // 4,
                StructureByteStride = 4,
                CounterOffsetInBytes = 0,
                Flags = native.D3D12_BUFFER_UAV_FLAGS.NONE
            )
        ),
        DestDescriptor = native.D3D12_CPU_DESCRIPTOR_HANDLE(cpu_handle.ptr + 1 * cbv_srv_uav_heap.descriptor_size),
        CounterResource = None
    )

device.CreateUnorderedAccessView(
        Resource=grads_buffer,
        Desc = native.D3D12_UNORDERED_ACCESS_VIEW_DESC(
            Format = native.DXGI_FORMAT.UNKNOWN,
            Buffer = native.D3D12_BUFFER_UAV(
                FirstElement = 0,
                NumElements = params_storage_size // 4,
                StructureByteStride = 4,
                CounterOffsetInBytes = 0,
                Flags = native.D3D12_BUFFER_UAV_FLAGS.NONE
            )
        ),
        DestDescriptor = native.D3D12_CPU_DESCRIPTOR_HANDLE(cpu_handle.ptr + 2 * cbv_srv_uav_heap.descriptor_size),
        CounterResource = None
    )

device.CreateShaderResourceView(
    Resource = target_texture,
    Desc = native.D3D12_SHADER_RESOURCE_VIEW_DESC(
        Format = target_texture.GetDesc().Format,
        Shader4ComponentMapping = native.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Texture2D = native.D3D12_TEX2D_SRV(
            MipLevels = 1,
            MostDetailedMip = 0,
            PlaneSlice = 0,
            ResourceMinLODClamp = 0.0
        )
    ),
    DestDescriptor = srv_cpu_handle
)

cmd_queue = device.CreateCommandQueue(native.D3D12_COMMAND_QUEUE_DESC(
    Type = native.D3D12_COMMAND_LIST_TYPE.DIRECT,
    Priority = 0,
    Flags = native.D3D12_COMMAND_QUEUE_FLAGS.NONE,
    NodeMask = 0
))

rdoc_start_capture()

class CBuffer(ctypes.Structure):
    _fields_ = [
        ("frame_idx", ctypes.c_uint32),
    ]

cbuffer = CBuffer()


cmd_alloc = device.CreateCommandAllocator(native.D3D12_COMMAND_LIST_TYPE.DIRECT)

for frame_idx in range(1 << 12):

    cmd_list = device.CreateCommandList(NodeMask=0, Type=native.D3D12_COMMAND_LIST_TYPE.DIRECT, Allocator=cmd_alloc)

    cmd_list.SetComputeRootSignature(signature)
    cmd_list.SetDescriptorHeaps([cbv_srv_uav_heap.heap])
    cmd_list.SetComputeRootDescriptorTable(RootParameterIndex=0, BaseDescriptor=gpu_handle)
    cmd_list.SetComputeRootDescriptorTable(RootParameterIndex=1, BaseDescriptor=srv_gpu_handle)
    cbuffer.frame_idx = frame_idx
    cmd_list.SetComputeRoot32BitConstants(RootParameterIndex=2, Num32BitValuesToSet=ctypes.sizeof(cbuffer) // 4, SrcData=ctypes.addressof(cbuffer), DestOffsetIn32BitValues=0)
    
    if frame_idx == 0:
        cmd_list.SetPipelineState(Initialize_pso)
        cmd_list.Dispatch(uav_texture.GetDesc().Width // 8, uav_texture.GetDesc().Height // 8, 1)
   
    cmd_list.ResourceBarrier([native.D3D12_RESOURCE_BARRIER(
        UAV = native.D3D12_RESOURCE_UAV_BARRIER(
            Resource = grads_buffer
        )
    )])

    cmd_list.SetPipelineState(Main_pso)
    cmd_list.Dispatch(1024, 1, 1)
    
    cmd_list.ResourceBarrier([native.D3D12_RESOURCE_BARRIER(
        UAV = native.D3D12_RESOURCE_UAV_BARRIER(
            Resource = grads_buffer
        )
    )])
    cmd_list.SetPipelineState(Backwards_pso)
    cmd_list.Dispatch(uav_texture.GetDesc().Width // 8, uav_texture.GetDesc().Height // 8, 1)

    cmd_list.ResourceBarrier([native.D3D12_RESOURCE_BARRIER(
        UAV = native.D3D12_RESOURCE_UAV_BARRIER(
            Resource = grads_buffer
        )
    )])
    cmd_list.SetPipelineState(Inference_pso)
    cmd_list.Dispatch(uav_texture.GetDesc().Width // 8, uav_texture.GetDesc().Height // 8, 1)

    cmd_list.Close()

    fence = device.CreateFence(0, native.D3D12_FENCE_FLAGS.NONE)
    e = native.Event()
    fence.SetEventOnCompletion(1, e)
    cmd_queue.ExecuteCommandLists([cmd_list])
    cmd_queue.Signal(fence, 1)
    e.Wait()

    print(f"Frame {frame_idx} completed.")

    if frame_idx % 16 == 0:
        dds = make_dds_from_texture(device, uav_texture)
        dds.save(get_or_create_tmp_folder() / "tmp.dds")


rdoc_end_capture()


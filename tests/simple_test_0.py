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
from py.dxc import *
import ctypes

launch_debugviewpp()

native = find_native_module("native")

debug = native.ID3D12Debug()
debug.EnableDebugLayer()

print(f"native.square(4) = {native.square(4)}")

assert native.square(4) == 16

factory = native.IDXGIFactory()
adapters = factory.EnumAdapters()

for adapter in adapters:
    print(f"Adapter: {adapter.GetDesc().Description}")
    pass

device = native.CreateDevice(adapters[0], native.D3D_FEATURE_LEVEL._11_0)

buffer = device.CreateCommittedResource(
    heapProperties = native.D3D12_HEAP_PROPERTIES(
        Type = native.D3D12_HEAP_TYPE.CUSTOM,
        CPUPageProperty = native.D3D12_CPU_PAGE_PROPERTY.WRITE_COMBINE,
        MemoryPoolPreference = native.D3D12_MEMORY_POOL.L0,
        CreationNodeMask = 1,
        VisibleNodeMask = 1
    ),
    heapFlags = native.D3D12_HEAP_FLAGS.NONE,
    resourceDesc = native.D3D12_RESOURCE_DESC(
        Dimension = native.D3D12_RESOURCE_DIMENSION.BUFFER,
        Alignment = 0,
        Width = 1024,
        Height = 1,
        DepthOrArraySize = 1,
        MipLevels = 1,
        Format = native.DXGI_FORMAT.UNKNOWN,
        SampleDesc = native.DXGI_SAMPLE_DESC(
            Count = 1,
            Quality = 0
        ),
        Layout = native.D3D12_TEXTURE_LAYOUT.ROW_MAJOR,
        Flags = native.D3D12_RESOURCE_FLAGS.NONE
    ),
    initialState = native.D3D12_RESOURCE_STATES.COPY_DEST,
    optimizedClearValue = None
)
print(f"Buffer virtual address: {hex(buffer.GetGPUVirtualAddress())}")
if 0:
    mapped_ptr = buffer.Map()
    array = (ctypes.c_float * 1024).from_address(mapped_ptr)
    for i in range(1024):
        array[i] = i

    buffer.Unmap()

    mapped_ptr = buffer.Map()
    array = (ctypes.c_float * 1024).from_address(mapped_ptr)
    for i in range(1024):
        assert array[i] == i

    buffer.Unmap()

print(CONSOLE_COLOR_GREEN, "Buffer Creation Test: SUCCESS", CONSOLE_COLOR_END)

dxc_ctx = DXCContext()
bytecode = dxc_ctx.compile_to_dxil(
    source = """
//js
#define ROOT_SIGNATURE_MACRO \
"UAV(u0, visibility = SHADER_VISIBILITY_ALL)," \

RWByteAddressBuffer u0 : register(u0);

[RootSignature(ROOT_SIGNATURE_MACRO)]
[numthreads(32, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    u0.Store<int>(DTid.x * sizeof(int), (int)123456);
}

//!js
""",
    args = "-E main -T cs_6_5",
)
assert bytecode is not None

print(CONSOLE_COLOR_GREEN, "DXC Test: SUCCESS", CONSOLE_COLOR_END)

signature = device.CreateRootSignature(
    Bytes = bytecode
)
assert signature is not None
pso_desc = native.D3D12_COMPUTE_PIPELINE_STATE_DESC(
    pRootSignature = signature,
    CS = native.D3D12_SHADER_BYTECODE(bytecode),
    NodeMask = 0,
    CachedPSO = None,
    Flags = native.D3D12_PIPELINE_STATE_FLAGS.NONE
)
pso = device.CreateComputePipelineState(pso_desc)

assert pso is not None

print(pso.GetISA())

print(CONSOLE_COLOR_GREEN, "Compute PSO Test: SUCCESS", CONSOLE_COLOR_END)

cmd_queue = device.CreateCommandQueue(native.D3D12_COMMAND_QUEUE_DESC(
    Type = native.D3D12_COMMAND_LIST_TYPE.DIRECT,
    Priority = 0,
    Flags = native.D3D12_COMMAND_QUEUE_FLAGS.NONE,
    NodeMask = 0
))

print(CONSOLE_COLOR_GREEN, "Command Queue Test: SUCCESS", CONSOLE_COLOR_END)

fence = device.CreateFence(0, native.D3D12_FENCE_FLAGS.NONE)

print(CONSOLE_COLOR_GREEN, "Fence Test: SUCCESS", CONSOLE_COLOR_END)

cmd_alloc = device.CreateCommandAllocator(native.D3D12_COMMAND_LIST_TYPE.DIRECT)

print(CONSOLE_COLOR_GREEN, "Command Allocator Test: SUCCESS", CONSOLE_COLOR_END)

cmd_list = device.CreateCommandList(NodeMask=0, Type=native.D3D12_COMMAND_LIST_TYPE.DIRECT, Allocator=cmd_alloc)

print(CONSOLE_COLOR_GREEN, "Command List Test: SUCCESS", CONSOLE_COLOR_END)

uav_buffer = device.CreateCommittedResource(
    heapProperties = native.D3D12_HEAP_PROPERTIES(
        Type = native.D3D12_HEAP_TYPE.DEFAULT,
        CPUPageProperty = native.D3D12_CPU_PAGE_PROPERTY.UNKNOWN,
        MemoryPoolPreference = native.D3D12_MEMORY_POOL.UNKNOWN,
        CreationNodeMask = 1,
        VisibleNodeMask = 1
    ),
    heapFlags = native.D3D12_HEAP_FLAGS.NONE,
    resourceDesc = native.D3D12_RESOURCE_DESC(
        Dimension = native.D3D12_RESOURCE_DIMENSION.BUFFER,
        Alignment = 0,
        Width = 1024,
        Height = 1,
        DepthOrArraySize = 1,
        MipLevels = 1,
        Format = native.DXGI_FORMAT.UNKNOWN,
        SampleDesc = native.DXGI_SAMPLE_DESC(
            Count = 1,
            Quality = 0
        ),
        Layout = native.D3D12_TEXTURE_LAYOUT.ROW_MAJOR,
        Flags = native.D3D12_RESOURCE_FLAGS.ALLOW_UNORDERED_ACCESS
    ),
    initialState = native.D3D12_RESOURCE_STATES.UNORDERED_ACCESS,
    optimizedClearValue = None
)

cmd_list.SetComputeRootSignature(signature)
cmd_list.SetPipelineState(pso)
cmd_list.SetComputeRootUnorderedAccessView(
    RootParameterIndex=0,
    BufferLocation=uav_buffer.GetGPUVirtualAddress()
)
cmd_list.Dispatch(1, 1, 1)

cmd_list.ResourceBarrier([
    native.D3D12_RESOURCE_BARRIER(
        Transition = native.D3D12_RESOURCE_TRANSITION_BARRIER(
            Resource = uav_buffer,
            Subresource = 0,
            StateBefore = native.D3D12_RESOURCE_STATES.UNORDERED_ACCESS,
            StateAfter = native.D3D12_RESOURCE_STATES.COPY_SOURCE
        )
    )
])

cmd_list.CopyBufferRegion(
    DestBuffer = buffer,
    DestOffset = 0,
    SrcBuffer = uav_buffer,
    SrcOffset = 0,
    NumBytes = 1024
)

cmd_list.ResourceBarrier([
    native.D3D12_RESOURCE_BARRIER(
        Transition = native.D3D12_RESOURCE_TRANSITION_BARRIER(
            Resource = uav_buffer,
            Subresource = 0,
            StateBefore = native.D3D12_RESOURCE_STATES.COPY_SOURCE,
            StateAfter = native.D3D12_RESOURCE_STATES.UNORDERED_ACCESS
        )
    )
])

cmd_list.Close()

e = native.Event()
fence.SetEventOnCompletion(1, e)
cmd_queue.ExecuteCommandLists([cmd_list])
cmd_queue.Signal(fence, 1)
e.Wait()

mapped_ptr = buffer.Map()
array = (ctypes.c_uint * 16).from_address(mapped_ptr)
print(f"Buffer contents: {array[:]}")

assert array[0] == 123456

buffer.Unmap()

print(CONSOLE_COLOR_GREEN, "SUCCESS", CONSOLE_COLOR_END)
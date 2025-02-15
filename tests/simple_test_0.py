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

native = find_native_module("native")

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
    initialState = native.D3D12_RESOURCE_STATES.COMMON,
    optimizedClearValue = None
)
print(f"Buffer virtual address: {hex(buffer.GetGPUVirtualAddress())}")

mapped_ptr = buffer.Map()
array = (ctypes.c_float * 1024).from_address(mapped_ptr)
for i in range(1024):
    array[i] = i

buffer.Unmap()

mapped_ptr = buffer.Map()
array = (ctypes.c_float * 1024).from_address(mapped_ptr)
for i in range(1024):
    assert array[i] == i


print(CONSOLE_COLOR_GREEN, "Buffer Creation Test: SUCCESS", CONSOLE_COLOR_END)

dxc_ctx = DXCContext()
bytecode = dxc_ctx.compile_to_dxil(
    source = """
//js
#define ROOT_SIGNATURE_MACRO \
"UAV(u0, visibility = SHADER_VISIBILITY_ALL)," \

[RootSignature(ROOT_SIGNATURE_MACRO)]
[numthreads(32, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{

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

print(CONSOLE_COLOR_GREEN, "Compute PSO Test: SUCCESS", CONSOLE_COLOR_END)

print(CONSOLE_COLOR_GREEN, "SUCCESS", CONSOLE_COLOR_END)
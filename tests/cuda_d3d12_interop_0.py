# MIT License
# Copyright (c) 2025 Anton Schreiner

import ctypes
from py.utils import *
from py.dxc import *

launch_debugviewpp()

native = find_native_module("native")

debug = native.ID3D12Debug()
debug.EnableDebugLayer()

factory = native.IDXGIFactory()
adapters = factory.EnumAdapters()

for adapter in adapters:
    print(f"Adapter: {adapter.GetDesc().Description}")

device = native.CreateDevice(adapters[0], native.D3D_FEATURE_LEVEL._11_0)

# Buffer size
BUFFER_SIZE = 256 * 4  # 256 floats

# Create a SHARED buffer that CUDA can access
shared_buffer = device.CreateCommittedResource(
    heapProperties=native.D3D12_HEAP_PROPERTIES(
        Type=native.D3D12_HEAP_TYPE.DEFAULT,
        CPUPageProperty=native.D3D12_CPU_PAGE_PROPERTY.UNKNOWN,
        MemoryPoolPreference=native.D3D12_MEMORY_POOL.UNKNOWN,
        CreationNodeMask=1,
        VisibleNodeMask=1
    ),
    heapFlags=native.D3D12_HEAP_FLAGS.SHARED,  # Key: must be SHARED for CUDA interop
    resourceDesc=native.D3D12_RESOURCE_DESC(
        Dimension=native.D3D12_RESOURCE_DIMENSION.BUFFER,
        Alignment=0,
        Width=BUFFER_SIZE,
        Height=1,
        DepthOrArraySize=1,
        MipLevels=1,
        Format=native.DXGI_FORMAT.UNKNOWN,
        SampleDesc=native.DXGI_SAMPLE_DESC(Count=1, Quality=0),
        Layout=native.D3D12_TEXTURE_LAYOUT.ROW_MAJOR,
        Flags=native.D3D12_RESOURCE_FLAGS.ALLOW_UNORDERED_ACCESS
    ),
    initialState=native.D3D12_RESOURCE_STATES.COMMON,
    optimizedClearValue=None
)

# Readback buffer (CPU-visible)
readback_buffer = device.CreateCommittedResource(
    heapProperties=native.D3D12_HEAP_PROPERTIES(
        Type=native.D3D12_HEAP_TYPE.READBACK,
        CPUPageProperty=native.D3D12_CPU_PAGE_PROPERTY.UNKNOWN,
        MemoryPoolPreference=native.D3D12_MEMORY_POOL.UNKNOWN,
        CreationNodeMask=1,
        VisibleNodeMask=1
    ),
    heapFlags=native.D3D12_HEAP_FLAGS.NONE,
    resourceDesc=native.D3D12_RESOURCE_DESC(
        Dimension=native.D3D12_RESOURCE_DIMENSION.BUFFER,
        Alignment=0,
        Width=BUFFER_SIZE,
        Height=1,
        DepthOrArraySize=1,
        MipLevels=1,
        Format=native.DXGI_FORMAT.UNKNOWN,
        SampleDesc=native.DXGI_SAMPLE_DESC(Count=1, Quality=0),
        Layout=native.D3D12_TEXTURE_LAYOUT.ROW_MAJOR,
        Flags=native.D3D12_RESOURCE_FLAGS.NONE
    ),
    initialState=native.D3D12_RESOURCE_STATES.COPY_DEST,
    optimizedClearValue=None
)

print(CONSOLE_COLOR_GREEN, "D3D12 Buffers Created", CONSOLE_COLOR_END)

# ============ CUDA Setup ============
from cuda.bindings import runtime as cudart
from cuda.bindings import driver as cuda

# Create CUDA stream
err, cuda_stream = cudart.cudaStreamCreate()
assert err == cudart.cudaError_t.cudaSuccess, f"cudaStreamCreate failed: {err}"

# Create shared fence for synchronization
fence = device.CreateFence(0, native.D3D12_FENCE_FLAGS.SHARED)
fence_value = 0

# Import fence into CUDA
sem_desc = cudart.cudaExternalSemaphoreHandleDesc()
sem_desc.type = cudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeD3D12Fence
sem_desc.handle.win32.handle = device.CreateSharedHandle(fence.GetRawPtr())
sem_desc.flags = 0

err, ext_sem = cudart.cudaImportExternalSemaphore(sem_desc)
assert err == cudart.cudaError_t.cudaSuccess, f"cudaImportExternalSemaphore failed: {err}"

print(CONSOLE_COLOR_GREEN, "CUDA Semaphore Imported", CONSOLE_COLOR_END)

# Import D3D12 buffer into CUDA
mem_desc = cudart.cudaExternalMemoryHandleDesc()
mem_desc.type = cudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D12Resource
mem_desc.handle.win32.handle = device.CreateSharedHandle(shared_buffer.GetRawPtr())
mem_desc.size = BUFFER_SIZE
mem_desc.flags = cudart.cudaExternalMemoryDedicated

err, ext_mem = cudart.cudaImportExternalMemory(mem_desc)
assert err == cudart.cudaError_t.cudaSuccess, f"cudaImportExternalMemory failed: {err}"

# Get mapped buffer pointer
buffer_desc = cudart.cudaExternalMemoryBufferDesc()
buffer_desc.offset = 0
buffer_desc.size = BUFFER_SIZE
buffer_desc.flags = 0

err, cuda_ptr = cudart.cudaExternalMemoryGetMappedBuffer(ext_mem, buffer_desc)
assert err == cudart.cudaError_t.cudaSuccess, f"cudaExternalMemoryGetMappedBuffer failed: {err}"

print(f"CUDA device pointer: {hex(cuda_ptr)}")
print(CONSOLE_COLOR_GREEN, "CUDA External Memory Mapped", CONSOLE_COLOR_END)

# ============ CUDA Kernel via torch inline compilation ============
import torch
from torch.utils.cpp_extension import load_inline

cuda_kernel_source = """
//js
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fill_pattern_kernel(float* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // Fill with a recognizable pattern: idx * 3.14159
        output[idx] = (float)idx * 3.14159f;
    }
}

void fill_pattern(uint64_t ptr, int num_elements, uint64_t stream) {
    float* output = reinterpret_cast<float*>(ptr);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    
    fill_pattern_kernel<<<blocks, threads, 0, cuda_stream>>>(output, num_elements);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
    }
}
;//
"""

cpp_source = """
void fill_pattern(uint64_t ptr, int num_elements, uint64_t stream);
"""

cuda_module = load_inline(
    name="d3d12_cuda_interop",
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel_source,
    functions=["fill_pattern"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["msvcrt.lib", "ucrt.lib"],
)

print(CONSOLE_COLOR_GREEN, "CUDA Kernel Compiled", CONSOLE_COLOR_END)

# ============ Execute ============

# D3D12 command queue setup
cmd_queue = device.CreateCommandQueue(native.D3D12_COMMAND_QUEUE_DESC(
    Type=native.D3D12_COMMAND_LIST_TYPE.DIRECT,
    Priority=0,
    Flags=native.D3D12_COMMAND_QUEUE_FLAGS.NONE,
    NodeMask=0
))
cmd_alloc = device.CreateCommandAllocator(native.D3D12_COMMAND_LIST_TYPE.DIRECT)
cmd_list = device.CreateCommandList(NodeMask=0, Type=native.D3D12_COMMAND_LIST_TYPE.DIRECT, Allocator=cmd_alloc)

# Signal from D3D12 that buffer is ready for CUDA
fence_value += 1
cmd_list.Close()
cmd_queue.ExecuteCommandLists([cmd_list])
cmd_queue.Signal(fence, fence_value)

# CUDA waits for D3D12
wait_params = cudart.cudaExternalSemaphoreWaitParams()
wait_params.params.fence.value = fence_value
err, = cudart.cudaWaitExternalSemaphoresAsync([ext_sem], [wait_params], 1, cuda_stream)
assert err == cudart.cudaError_t.cudaSuccess, f"cudaWaitExternalSemaphoresAsync failed: {err}"

# Run CUDA kernel
num_floats = BUFFER_SIZE // 4
cuda_module.fill_pattern(cuda_ptr, num_floats, int(cuda_stream))

# CUDA signals completion
fence_value += 1
signal_params = cudart.cudaExternalSemaphoreSignalParams()
signal_params.params.fence.value = fence_value
err, = cudart.cudaSignalExternalSemaphoresAsync([ext_sem], [signal_params], 1, cuda_stream)
assert err == cudart.cudaError_t.cudaSuccess, f"cudaSignalExternalSemaphoresAsync failed: {err}"

print(CONSOLE_COLOR_GREEN, "CUDA Kernel Executed", CONSOLE_COLOR_END)

# ============ D3D12 Readback ============

# Reset command list
cmd_alloc.Reset()
cmd_list.Reset(cmd_alloc, None)

# Transition and copy
cmd_list.ResourceBarrier([
    native.D3D12_RESOURCE_BARRIER(
        Transition=native.D3D12_RESOURCE_TRANSITION_BARRIER(
            Resource=shared_buffer,
            Subresource=0,
            StateBefore=native.D3D12_RESOURCE_STATES.COMMON,
            StateAfter=native.D3D12_RESOURCE_STATES.COPY_SOURCE
        )
    )
])

cmd_list.CopyBufferRegion(
    DstBuffer=readback_buffer,
    DstOffset=0,
    SrcBuffer=shared_buffer,
    SrcOffset=0,
    NumBytes=BUFFER_SIZE
)

cmd_list.Close()

# Wait for CUDA fence, then execute copy
e = native.Event()
fence.SetEventOnCompletion(fence_value, e)
e.Wait()  # Wait for CUDA to finish

fence_value += 1
cmd_queue.ExecuteCommandLists([cmd_list])
cmd_queue.Signal(fence, fence_value)
fence.SetEventOnCompletion(fence_value, e)
e.Wait()

# Read results
mapped_ptr = readback_buffer.Map()
result_array = (ctypes.c_float * num_floats).from_address(mapped_ptr)

print("\n" + "=" * 50)
print("Buffer contents (first 16 floats):")
for i in range(16):
    expected = i * 3.14159
    actual = result_array[i]
    match = "✓" if abs(actual - expected) < 0.001 else "✗"
    print(f"  [{i:2d}] = {actual:10.5f}  (expected {expected:10.5f}) {match}")

print("=" * 50)

# Verify all values
all_correct = all(abs(result_array[i] - i * 3.14159) < 0.001 for i in range(num_floats))
readback_buffer.Unmap()

if all_correct:
    print(CONSOLE_COLOR_GREEN, "\nSUCCESS: CUDA modified D3D12 buffer correctly!", CONSOLE_COLOR_END)
else:
    print(CONSOLE_COLOR_RED, "\nFAILED: Values don't match expected pattern", CONSOLE_COLOR_END)

# Cleanup
cudart.cudaDestroyExternalSemaphore(ext_sem)
cudart.cudaDestroyExternalMemory(ext_mem)
cudart.cudaStreamDestroy(cuda_stream)
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

if args.load_pix:
    pix.PIXLoadLatestWinPixGpuCapturerLibrary()
    # pix.PIXBeginCapture()

if args.load_rdoc and  find_rdoc() is not None:
    rdoc_load()

native = find_native_module("native")

debug = native.ID3D12Debug()
debug.EnableDebugLayer()
# debug.SetEnableGPUBasedValidation(True)

print(f"native.square(4) = {native.square(4)}")

assert native.square(4) == 16

factory = native.IDXGIFactory()
adapters = factory.EnumAdapters()
device = native.CreateDevice(adapters[0], native.D3D_FEATURE_LEVEL._11_0)

rdoc_start_capture()

dxc_ctx = DXCContext()
bytecode = dxc_ctx.compile_to_dxil(
    source = """
//js
#define ROOT_SIGNATURE_MACRO \
 "DescriptorTable(" \
            "UAV(u0, NumDescriptors = 1, flags = DESCRIPTORS_VOLATILE) " \
            "), " \
            "SRV(t0), " \
            "StaticSampler(s0, " \
                "Filter = FILTER_MIN_MAG_MIP_LINEAR, " \
                "AddressU = TEXTURE_ADDRESS_WRAP, " \
                "AddressV = TEXTURE_ADDRESS_WRAP, " \
                "AddressW = TEXTURE_ADDRESS_WRAP), " \
            "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), " \


RWTexture2D<float4> u0 : register(u0);

RaytracingAccelerationStructure tlas : register(t0);

[RootSignature(ROOT_SIGNATURE_MACRO)]
[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint2 dims;
    u0.GetDimensions(dims.x, dims.y);
    float2 uv = (float2(DTid.xy) + 0.5f) / float2(dims.xy); 

    float t = 3.14159265359f * 0.333f;
    float camera_radius     = 4.0f;
    float3 camera_pos       = float3(sin(t) * camera_radius, 0.0f, cos(t) * camera_radius);
    float3 camera_look_at   = float3(0.0f, 0.0f, 0.0f);
    float3 camera_look      = normalize(camera_look_at - camera_pos);
    float3 camera_up        = float3(0.0f, 1.0f, 0.0f);
    float3 camera_right     = normalize(cross(camera_look, camera_up));
    camera_up               = normalize(cross(camera_look, camera_right));

    float3 ray_origin       = camera_pos;
    float fov               = 3.14159265359f / 2.0f;
    float aspect_ratio      = 1.0f;
    float fov_tan           = tan(fov * 0.5f);
    float2 ndc              = uv.xy * 2.0f - 1.0f;
    float3 ray_direction    = normalize(camera_look + ndc.x * camera_right * aspect_ratio * fov_tan + ndc.y * camera_up * fov_tan);


    RayDesc ray_desc                = (RayDesc)0;
    ray_desc.Direction              = ray_direction;
    ray_desc.Origin                 = ray_origin;
    ray_desc.TMin                   = float(0.0);
    ray_desc.TMax                   = float(1.0e6);
    RayQuery<RAY_FLAG_NONE> ray_query;
    ray_query.TraceRayInline(tlas, RAY_FLAG_NONE, 0xffu, ray_desc);
    ray_query.Proceed();
    if (ray_query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float2 bary = ray_query.CommittedTriangleBarycentrics();
        u0[DTid.xy] = float4(bary.xy, 0.0f, 1.0f);
    } else {
        u0[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
  
}

//!js
""",
    args = "-E main -T cs_6_5",
)
assert bytecode is not None

signature = device.CreateRootSignature(
    Bytes = bytecode
)
assert signature is not None
pso_desc = native.D3D12_COMPUTE_PIPELINE_STATE_DESC(
    RootSignature = signature,
    CS = native.D3D12_SHADER_BYTECODE(bytecode),
    NodeMask = 0,
    CachedPSO = None,
    Flags = native.D3D12_PIPELINE_STATE_FLAGS.NONE
)
pso = device.CreateComputePipelineState(pso_desc)

assert pso is not None

width, height = 512, 512

class Vertex(ctypes.Structure):
    _fields_ = [
        ("position", ctypes.c_float * 3),
    ]

cube_indices = [
    0, 1, 2, 0, 2, 3,  # front face
    4, 5, 6, 4, 6, 7,  # back face
    0, 1, 5, 0, 5, 4,  # bottom face
    2, 3, 7, 2, 7, 6,  # top face
    0, 3, 7, 0, 7, 4,  # left face
    1, 2, 6, 1, 6, 5   # right face
]

cube_vertices = [
    Vertex(position=(-1.0, -1.0, -1.0)),
    Vertex(position=( 1.0, -1.0, -1.0)),
    Vertex(position=( 1.0,  1.0, -1.0)),
    Vertex(position=(-1.0,  1.0, -1.0)),
    Vertex(position=(-1.0, -1.0,  1.0)),
    Vertex(position=( 1.0, -1.0,  1.0)),
    Vertex(position=( 1.0,  1.0,  1.0)),
    Vertex(position=(-1.0,  1.0,  1.0)),

]

vertex_buffer = make_write_combined_buffer(device, size=len(cube_vertices) * ctypes.sizeof(Vertex))
if 1: # Upload
    mapped_ptr = vertex_buffer.Map()
    array = (Vertex * len(cube_vertices)).from_address(mapped_ptr)
    for i in range(len(cube_vertices)):
        array[i] = cube_vertices[i]
    vertex_buffer.Unmap()

index_buffer = make_write_combined_buffer(device, size=len(cube_indices) * ctypes.sizeof(ctypes.c_uint32))
if 1: # Upload
    mapped_ptr = index_buffer.Map()
    array = (ctypes.c_uint32 * len(cube_indices)).from_address(mapped_ptr)
    for i in range(len(cube_indices)):
        array[i] = cube_indices[i]
    index_buffer.Unmap()

as_inputs = native.D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS(
    Flags = native.D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS.PREFER_FAST_TRACE | native.D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS.ALLOW_UPDATE,
    GeometryDescs = [
        native.D3D12_RAYTRACING_GEOMETRY_DESC(
            Flags = native.D3D12_RAYTRACING_GEOMETRY_FLAGS.OPAQUE,
            Triangles = native.D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC(
                Transform3x4               = 0,
                IndexFormat                = native.DXGI_FORMAT.R32_UINT,
                VertexFormat               = native.DXGI_FORMAT.R32G32B32_FLOAT,
                IndexCount                 = len(cube_indices),
                VertexCount                = len(cube_vertices),
                IndexBuffer                = index_buffer.GetGPUVirtualAddress(),
                VertexBuffer               = native.D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE(
                    StartAddress             = vertex_buffer.GetGPUVirtualAddress(),
                    StrideInBytes            = ctypes.sizeof(Vertex)
                ),
            )
        )
    ]
)

prebuild_info = device.GetRaytracingAccelerationStructurePrebuildInfo(as_inputs)
print(f"ResultDataMaxSizeInBytes: {prebuild_info.ResultDataMaxSizeInBytes}")
print(f"ScratchDataSizeInBytes: {prebuild_info.ScratchDataSizeInBytes}")
print(f"UpdateScratchDataSizeInBytes: {prebuild_info.UpdateScratchDataSizeInBytes}")

blas_result_buffer  = make_uav_buffer(device, prebuild_info.ResultDataMaxSizeInBytes, state=native.D3D12_RESOURCE_STATES.RAYTRACING_ACCELERATION_STRUCTURE)
blas_scratch_buffer = make_uav_buffer(device, prebuild_info.ScratchDataSizeInBytes)

instances = [
    D3D12_RAYTRACING_INSTANCE_DESC(
        Transform = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
        ),
        InstanceID = 0,
        InstanceMask = 0xFF,
        InstanceContributionToHitGroupIndex = 0,
        Flags = native.D3D12_RAYTRACING_INSTANCE_FLAGS.NONE,
        AccelerationStructure = blas_result_buffer.GetGPUVirtualAddress(),
    )
]

instance_buffer = make_write_combined_buffer(device, size=len(instances) * ctypes.sizeof(D3D12_RAYTRACING_INSTANCE_DESC))
if 1: # Upload
    mapped_ptr = instance_buffer.Map()
    array = (D3D12_RAYTRACING_INSTANCE_DESC * len(instances)).from_address(mapped_ptr)
    for i in range(len(instances)):
        array[i] = instances[i]
    instance_buffer.Unmap()

tlas_inputs = native.D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS(
    Flags = native.D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS.PREFER_FAST_TRACE | native.D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS.ALLOW_UPDATE,
    NumDescs = 1,
    InstanceDescs = instance_buffer.GetGPUVirtualAddress()
)
tlas_prebuild_info = device.GetRaytracingAccelerationStructurePrebuildInfo(tlas_inputs)
print(f"ResultDataMaxSizeInBytes: {tlas_prebuild_info.ResultDataMaxSizeInBytes}")
print(f"ScratchDataSizeInBytes: {tlas_prebuild_info.ScratchDataSizeInBytes}")
print(f"UpdateScratchDataSizeInBytes: {tlas_prebuild_info.UpdateScratchDataSizeInBytes}")

tlas_result_buffer  = make_uav_buffer(device, tlas_prebuild_info.ResultDataMaxSizeInBytes, state=native.D3D12_RESOURCE_STATES.RAYTRACING_ACCELERATION_STRUCTURE)

tlas_build_info = native.D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC(
    Dest = tlas_result_buffer.GetGPUVirtualAddress(),
    Source = 0,
    Scratch = blas_scratch_buffer.GetGPUVirtualAddress(),
    Inputs = tlas_inputs,
)

blas_build_info = native.D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC(
    Dest = blas_result_buffer.GetGPUVirtualAddress(),
    Source = 0,
    Scratch = blas_scratch_buffer.GetGPUVirtualAddress(),
    Inputs = as_inputs,
)

uav_texture = device.CreateCommittedResource(
    heapProperties = native.D3D12_HEAP_PROPERTIES(
        Type = native.D3D12_HEAP_TYPE.DEFAULT,
        CPUPageProperty = native.D3D12_CPU_PAGE_PROPERTY.UNKNOWN,
        MemoryPoolPreference = native.D3D12_MEMORY_POOL.UNKNOWN,
        CreationNodeMask = 1,
        VisibleNodeMask = 1
    ),
    heapFlags = native.D3D12_HEAP_FLAGS.NONE,
    resourceDesc = native.D3D12_RESOURCE_DESC(
        Dimension = native.D3D12_RESOURCE_DIMENSION.TEXTURE2D,
        Alignment = 0,
        Width = width,
        Height = height,
        DepthOrArraySize = 1,
        MipLevels = 1,
        Format = native.DXGI_FORMAT.R16G16B16A16_FLOAT,
        SampleDesc = native.DXGI_SAMPLE_DESC(
            Count = 1,
            Quality = 0
        ),
        Layout = native.D3D12_TEXTURE_LAYOUT.UNKNOWN,
        Flags = native.D3D12_RESOURCE_FLAGS.ALLOW_UNORDERED_ACCESS
    ),
    initialState = native.D3D12_RESOURCE_STATES.UNORDERED_ACCESS,
    optimizedClearValue = None
)

cbv_srv_uav_heap = CBV_SRV_UAV_DescriptorHeap(device, 1024)

cpu_handle, gpu_handle = cbv_srv_uav_heap.get_next_descriptor_handle()

device.CreateUnorderedAccessView(
    Resource = uav_texture,
    Desc = native.D3D12_UNORDERED_ACCESS_VIEW_DESC(
        Format = native.DXGI_FORMAT.R16G16B16A16_FLOAT,
        Texture2D = native.D3D12_TEX2D_UAV(
            MipSlice = 0,
            PlaneSlice = 0
        )
    ),
    DestDescriptor = cpu_handle,
    CounterResource = None
)
texture_gpu_descritpor = gpu_handle

cmd_queue = device.CreateCommandQueue(native.D3D12_COMMAND_QUEUE_DESC(
    Type = native.D3D12_COMMAND_LIST_TYPE.DIRECT,
    Priority = 0,
    Flags = native.D3D12_COMMAND_QUEUE_FLAGS.NONE,
    NodeMask = 0
))
fence = device.CreateFence(0, native.D3D12_FENCE_FLAGS.NONE)
cmd_alloc = device.CreateCommandAllocator(native.D3D12_COMMAND_LIST_TYPE.DIRECT)


cmd_list = device.CreateCommandList(NodeMask=0, Type=native.D3D12_COMMAND_LIST_TYPE.DIRECT, Allocator=cmd_alloc)

cmd_list.BuildRaytracingAccelerationStructure(blas_build_info)
cmd_list.ResourceBarrier([native.D3D12_RESOURCE_BARRIER(
    UAV = native.D3D12_RESOURCE_UAV_BARRIER(
        Resource = blas_scratch_buffer
    )
)])
cmd_list.BuildRaytracingAccelerationStructure(tlas_build_info)

cmd_list.Close()

e = native.Event()
fence.SetEventOnCompletion(1, e)
cmd_queue.ExecuteCommandLists([cmd_list])
cmd_queue.Signal(fence, 1)
e.Wait()

def render():
    cmd_list = device.CreateCommandList(NodeMask=0, Type=native.D3D12_COMMAND_LIST_TYPE.DIRECT, Allocator=cmd_alloc)

    cmd_list.SetPipelineState(pso)
    cmd_list.SetComputeRootSignature(signature)
    cmd_list.SetDescriptorHeaps([cbv_srv_uav_heap.heap])
    cmd_list.SetComputeRootDescriptorTable(
        RootParameterIndex  = 0,
        BaseDescriptor      = texture_gpu_descritpor
    )
    cmd_list.SetComputeRootShaderResourceView(
        RootParameterIndex  = 1,
        BufferLocation      = tlas_result_buffer.GetGPUVirtualAddress()
    )

    cmd_list.Dispatch(width // 8, height // 8, 1)


    cmd_list.Close()

    e = native.Event()
    fence.SetEventOnCompletion(1, e)
    cmd_queue.ExecuteCommandLists([cmd_list])
    cmd_queue.Signal(fence, 1)
    e.Wait()


if args.make_window:

    import PyQt5.QtWidgets as qtw
    from PyQt5.QtCore import Qt, QObject, QTimer, QEvent
    from PyQt5.QtGui import QSurfaceFormat

    class MainWindow:
        def __init__(self):
            self.window = qtw.QMainWindow()
            self.window.setWindowTitle("Simple Window")
            self.window.setGeometry(100, 100, 800, 600)
            self.window.show()
            self.hwnd = int(self.window.winId())
            self.swapchain = factory.CreateSwapChain(
                cmd_queue,
                native.DXGI_SWAP_CHAIN_DESC(
                    BufferDesc = native.DXGI_MODE_DESC(
                        Width = 512,
                        Height = 512,
                        RefreshRate = native.DXGI_RATIONAL(
                            Numerator = 0,
                            Denominator = 1
                        ),
                        Format = native.DXGI_FORMAT.R8G8B8A8_UNORM,
                        ScanlineOrdering = native.DXGI_MODE_SCANLINE_ORDER.UNSPECIFIED,
                        Scaling = native.DXGI_MODE_SCALING.UNSPECIFIED
                    ),
                    SampleDesc = native.DXGI_SAMPLE_DESC(
                        Count = 1,
                        Quality = 0
                    ),
                    BufferUsage = native.DXGI_USAGE.RENDER_TARGET_OUTPUT | native.DXGI_USAGE.SHADER_INPUT,
                    BufferCount = 2,
                    OutputWindow = self.hwnd,
                    Windowed = True,
                    SwapEffect = native.DXGI_SWAP_EFFECT.FLIP_DISCARD,
                    Flags = native.DXGI_SWAP_CHAIN_FLAG.FRAME_LATENCY_WAITABLE_OBJECT
                )
            )

            self.render_timer = QTimer()
            self.render_timer.timeout.connect(self.on_frame)
            self.render_timer.start(16)


        def on_frame(self):
            render()

            self.swapchain.Present(0, 0)

    app = qtw.QApplication([])
    window = MainWindow()
    app.exec_()

else:
    render()

rdoc_end_capture()

    # if args.load_pix:
        # pix.PIXEndCapture()


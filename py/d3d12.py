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

from .utils import *
from .dds import *
import ctypes

native = find_native_module("native")

class D3D12_RAYTRACING_INSTANCE_DESC(ctypes.Structure):
    _fields_ = [
        ("Transform", ctypes.c_float * 4 * 3),
        ("InstanceID", ctypes.c_uint32, 24),
        ("InstanceMask", ctypes.c_uint32, 8),
        ("InstanceContributionToHitGroupIndex", ctypes.c_uint32, 24),
        ("Flags", ctypes.c_uint32, 8),
        ("AccelerationStructure", ctypes.c_uint64)
    ]

assert ctypes.sizeof(D3D12_RAYTRACING_INSTANCE_DESC) == 64, "D3D12_RAYTRACING_INSTANCE_DESC is not the correct size"

class CBV_SRV_UAV_DescriptorHeap:
    def __init__(self, device, num_descriptors = 1 << 10):
        self.heap = device.CreateDescriptorHeap(native.D3D12_DESCRIPTOR_HEAP_DESC(
            Type = native.D3D12_DESCRIPTOR_HEAP_TYPE.CBV_SRV_UAV,
            NumDescriptors = num_descriptors,
            Flags = native.D3D12_DESCRIPTOR_HEAP_FLAGS.SHADER_VISIBLE,
            NodeMask = 0
        ))
        self.device = device
        self.num_descriptors = num_descriptors
        self.descriptor_size = device.GetDescriptorHandleIncrementSize(native.D3D12_DESCRIPTOR_HEAP_TYPE.CBV_SRV_UAV)
        self.next_descriptor = 0

    def get_cpu_descriptor_handle_for_index(self, index):
        return native.D3D12_CPU_DESCRIPTOR_HANDLE(self.heap.GetCPUDescriptorHandleForHeapStart().ptr + index * self.descriptor_size)

    def get_gpu_descriptor_handle_for_index(self, index):
        return native.D3D12_GPU_DESCRIPTOR_HANDLE(self.heap.GetGPUDescriptorHandleForHeapStart().ptr + index * self.descriptor_size)

    def get_cpu_descriptor_handle(self):
        return self.get_cpu_descriptor_handle_for_index(self.next_descriptor)

    def get_gpu_descriptor_handle(self):
        return self.get_gpu_descriptor_handle_for_index(self.next_descriptor)

    def get_next_descriptor_handle(self):
        cpu_handle, gpu_handle = self.get_cpu_descriptor_handle(), self.get_gpu_descriptor_handle()
        self.next_descriptor = (self.next_descriptor + 1) % self.num_descriptors
        return cpu_handle, gpu_handle

    def reset(self):
        self.next_descriptor = 0

def make_write_combined_buffer(device, size, state = native.D3D12_RESOURCE_STATES.GENERIC_READ, name = None):
    res = device.CreateCommittedResource(
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
                Width = size,
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
            initialState = state,
            optimizedClearValue = None
        )
    if name is not None:
        res.SetName(name)
    return res

def make_uav_buffer(device, size, state = native.D3D12_RESOURCE_STATES.UNORDERED_ACCESS, name = None):
    res = device.CreateCommittedResource(
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
                Width = size,
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
            initialState = state,
            optimizedClearValue = None
        )
    if name is not None:
        res.SetName(name)
    return res

def make_texture_from_dds(device, dds : DDSTexture):
    assert dds.dx10_header.resource_dimension == D3D10_RESOURCE_DIMENSION.TEXTURE2D.value, "Only 2D textures are supported"
    dst_tex = device.CreateCommittedResource(
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
                Alignment               = 0,
                Width                   = dds.header.width,
                Height                  = dds.header.height,
                DepthOrArraySize        = 1,
                MipLevels               = dds.header.mip_map_count,
                Format                  = native.DXGI_FORMAT(dds.dx10_header.dxgi_format),
                SampleDesc              = native.DXGI_SAMPLE_DESC(
                    Count = 1,
                    Quality = 0
                ),
                Layout = native.D3D12_TEXTURE_LAYOUT.UNKNOWN,
                Flags = native.D3D12_RESOURCE_FLAGS.NONE
            ),
            initialState = native.D3D12_RESOURCE_STATES.COPY_DEST,
            optimizedClearValue = None
        )
    desc = dst_tex.GetDesc()

    copyable_footprints = device.GetCopyableFootprints(Resource=desc,
                                                       FirstSubresource=0,
                                                       NumSubresources=dds.header.mip_map_count,
                                                       BaseOffset=0)
    upload_buffer = make_write_combined_buffer(device, sum([b for b in copyable_footprints.TotalBytes]))

    dst_data_ptr        = upload_buffer.Map(0, None)
    src_data_ptr        = dds.buf_ref.ptr
    src_bpp             = dds_get_bytes_per_pixel(DXGI_FORMAT(dds.dx10_header.dxgi_format))
    src_offset          = 0
    dst_offset          = 0

    cmd_queue = device.CreateCommandQueue(native.D3D12_COMMAND_QUEUE_DESC(
        Type = native.D3D12_COMMAND_LIST_TYPE.DIRECT,
        Priority = 0,
        Flags = native.D3D12_COMMAND_QUEUE_FLAGS.NONE,
        NodeMask = 0
    ))
    fence = device.CreateFence(0, native.D3D12_FENCE_FLAGS.NONE)
    cmd_alloc = device.CreateCommandAllocator(native.D3D12_COMMAND_LIST_TYPE.DIRECT)
    cmd_list = device.CreateCommandList(NodeMask=0, Type=native.D3D12_COMMAND_LIST_TYPE.DIRECT, Allocator=cmd_alloc)

    for array_idx in range(dds.dx10_header.array_size):
        for mip_idx in range(dds.header.mip_map_count):
            cur_width       = max(1, dds.header.width >> mip_idx)
            src_pitch       = src_bpp * cur_width
            subresource_idx = mip_idx + array_idx * dds.header.mip_map_count
            dst_offset      = copyable_footprints.Layouts[subresource_idx].Offset
            footprint       = copyable_footprints.Layouts[subresource_idx].Footprint
            num_rows        = copyable_footprints.NumRows[subresource_idx]
            # Copy the data from the source buffer to the upload buffer
            for row in range(num_rows):
                src_row_ptr = src_data_ptr + src_offset
                dst_row_ptr = dst_data_ptr + dst_offset
                ctypes.memmove(dst_row_ptr, src_row_ptr, src_pitch)
                src_offset += src_pitch
                dst_offset += footprint.RowPitch
            
            # Copy the data from the upload buffer to the destination texture
            dst_tex_loc = native.D3D12_TEXTURE_COPY_LOCATION(
                Resource = dst_tex,
                SubresourceIndex = subresource_idx
            )
            src_tex_loc = native.D3D12_TEXTURE_COPY_LOCATION(
                Resource = upload_buffer,
                PlacedFootprint = copyable_footprints.Layouts[subresource_idx]
            )
            cmd_list.CopyTextureRegion(dst_tex_loc, 0, 0, 0, src_tex_loc, None)

    cmd_list.ResourceBarrier([
        native.D3D12_RESOURCE_BARRIER(
            Transition = native.D3D12_RESOURCE_TRANSITION_BARRIER(
                Resource = dst_tex,
                Subresource = 0,
                StateBefore = native.D3D12_RESOURCE_STATES.COPY_DEST,
                StateAfter = native.D3D12_RESOURCE_STATES.PIXEL_SHADER_RESOURCE | native.D3D12_RESOURCE_STATES.NON_PIXEL_SHADER_RESOURCE
            )
        )
    ])

    cmd_list.Close()

    e = native.Event()
    fence.SetEventOnCompletion(1, e)
    cmd_queue.ExecuteCommandLists([cmd_list])
    cmd_queue.Signal(fence, 1)
    e.Wait()

    del upload_buffer

    return dst_tex
    

def make_texture_from_numpy_array_NHWC(device, arr : np.ndarray):
    
    if arr.dtype == np.uint8:
        src_bpp = 4
        dxgi_format = native.DXGI_FORMAT.R8G8B8A8_UNORM
    elif arr.dtype == np.float32:
        src_bpp = 16
        dxgi_format = native.DXGI_FORMAT.R32G32B32A32_FLOAT
    else:
        raise ValueError("Unsupported array dtype")
    
    batch_size, height, width, channels = arr.shape
    assert channels == 4, "Only 4 channel textures are supported"
    assert batch_size == 1, "Only single batch textures are supported"

    dst_tex = device.CreateCommittedResource(
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
                Alignment               = 0,
                Width                   = width,
                Height                  = height,
                DepthOrArraySize        = 1,
                MipLevels               = 1,
                Format                  = dxgi_format,
                SampleDesc              = native.DXGI_SAMPLE_DESC(
                    Count = 1,
                    Quality = 0
                ),
                Layout = native.D3D12_TEXTURE_LAYOUT.UNKNOWN,
                Flags = native.D3D12_RESOURCE_FLAGS.NONE
            ),
            initialState = native.D3D12_RESOURCE_STATES.COPY_DEST,
            optimizedClearValue = None
        )
    desc = dst_tex.GetDesc()

    copyable_footprints = device.GetCopyableFootprints(Resource=desc,
                                                       FirstSubresource=0,
                                                       NumSubresources=1,
                                                       BaseOffset=0)
    upload_buffer = make_write_combined_buffer(device, sum([b for b in copyable_footprints.TotalBytes]))

    dst_data_ptr        = upload_buffer.Map(0, None)
    src_data_ptr        = arr.ctypes.data
    src_offset          = 0
    dst_offset          = 0

    cmd_queue = device.CreateCommandQueue(native.D3D12_COMMAND_QUEUE_DESC(
        Type = native.D3D12_COMMAND_LIST_TYPE.DIRECT,
        Priority = 0,
        Flags = native.D3D12_COMMAND_QUEUE_FLAGS.NONE,
        NodeMask = 0
    ))
    fence = device.CreateFence(0, native.D3D12_FENCE_FLAGS.NONE)
    cmd_alloc = device.CreateCommandAllocator(native.D3D12_COMMAND_LIST_TYPE.DIRECT)
    cmd_list = device.CreateCommandList(NodeMask=0, Type=native.D3D12_COMMAND_LIST_TYPE.DIRECT, Allocator=cmd_alloc)

    for array_idx in range(1):
        for mip_idx in range(1):
            cur_width       = max(1, width >> mip_idx)
            src_pitch       = src_bpp * cur_width
            subresource_idx = mip_idx + array_idx * 1
            dst_offset      = copyable_footprints.Layouts[subresource_idx].Offset
            footprint       = copyable_footprints.Layouts[subresource_idx].Footprint
            num_rows        = copyable_footprints.NumRows[subresource_idx]
            # Copy the data from the source buffer to the upload buffer
            for row in range(num_rows):
                src_row_ptr = src_data_ptr + src_offset
                dst_row_ptr = dst_data_ptr + dst_offset
                ctypes.memmove(dst_row_ptr, src_row_ptr, src_pitch)
                src_offset += src_pitch
                dst_offset += footprint.RowPitch
            
            # Copy the data from the upload buffer to the destination texture
            dst_tex_loc = native.D3D12_TEXTURE_COPY_LOCATION(
                Resource = dst_tex,
                SubresourceIndex = subresource_idx
            )
            src_tex_loc = native.D3D12_TEXTURE_COPY_LOCATION(
                Resource = upload_buffer,
                PlacedFootprint = copyable_footprints.Layouts[subresource_idx]
            )
            cmd_list.CopyTextureRegion(dst_tex_loc, 0, 0, 0, src_tex_loc, None)

    cmd_list.ResourceBarrier([
        native.D3D12_RESOURCE_BARRIER(
            Transition = native.D3D12_RESOURCE_TRANSITION_BARRIER(
                Resource = dst_tex,
                Subresource = 0,
                StateBefore = native.D3D12_RESOURCE_STATES.COPY_DEST,
                StateAfter = native.D3D12_RESOURCE_STATES.PIXEL_SHADER_RESOURCE | native.D3D12_RESOURCE_STATES.NON_PIXEL_SHADER_RESOURCE
            )
        )
    ])

    cmd_list.Close()

    e = native.Event()
    fence.SetEventOnCompletion(1, e)
    cmd_queue.ExecuteCommandLists([cmd_list])
    cmd_queue.Signal(fence, 1)
    e.Wait()

    del upload_buffer

    return dst_tex
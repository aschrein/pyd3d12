/*
MIT License

Copyright (c) 2025 Anton Schreiner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "common.h"

#include <d3dx12/d3dx12.h>
#include <dxgi1_6.h>
#include <optional>
#include <sstream>

namespace py = pybind11;
using namespace pybind11::literals;

float square(float x) { return x * x; }

class DXGI_ADAPTER_DESC_WRAPPER {
public:
    std::wstring Description;
    UINT         VendorId;
    UINT         DeviceId;
    UINT         SubSysId;
    UINT         Revision;
    SIZE_T       DedicatedVideoMemory;
    SIZE_T       DedicatedSystemMemory;
    SIZE_T       SharedSystemMemory;
    LUID         AdapterLuid;

public:
    DXGI_ADAPTER_DESC_WRAPPER(DXGI_ADAPTER_DESC desc) {
        Description           = desc.Description;
        VendorId              = desc.VendorId;
        DeviceId              = desc.DeviceId;
        SubSysId              = desc.SubSysId;
        Revision              = desc.Revision;
        DedicatedVideoMemory  = desc.DedicatedVideoMemory;
        DedicatedSystemMemory = desc.DedicatedSystemMemory;
        SharedSystemMemory    = desc.SharedSystemMemory;
        AdapterLuid           = desc.AdapterLuid;
    }

    DXGI_ADAPTER_DESC ToNative() {
        DXGI_ADAPTER_DESC desc = {};
        // desc.Description           = Description.c_str();
        memcpy_s(desc.Description, sizeof(desc.Description), Description.c_str(), Description.size() * sizeof(wchar_t));
        desc.VendorId              = VendorId;
        desc.DeviceId              = DeviceId;
        desc.SubSysId              = SubSysId;
        desc.Revision              = Revision;
        desc.DedicatedVideoMemory  = DedicatedVideoMemory;
        desc.DedicatedSystemMemory = DedicatedSystemMemory;
        desc.SharedSystemMemory    = SharedSystemMemory;
        desc.AdapterLuid           = AdapterLuid;
        return desc;
    }
};
class EventWrapper {
public:
    HANDLE event = nullptr;

public:
    EventWrapper() { event = CreateEvent(nullptr, FALSE, FALSE, nullptr); }

    ~EventWrapper() {
        if (event) CloseHandle(event);
    }

    HANDLE Get() { return event; }
    void   Set() { SetEvent(event); }
    void   Reset() { ResetEvent(event); }
    void   Wait() { WaitForSingleObject(event, INFINITE); }
};

class ID3D12FenceWrapper {
public:
    ID3D12Fence * fence  = nullptr;
    ID3D12Fence1 *fence1 = nullptr;

public:
    ID3D12FenceWrapper(ID3D12Fence *fence) : fence(fence) { fence->QueryInterface(IID_PPV_ARGS(&fence1)); }

    ~ID3D12FenceWrapper() {
        if (fence) fence->Release();
        if (fence1) fence1->Release();
    }

    void   Signal(UINT64 value) { fence->Signal(value); }
    UINT64 GetCompletedValue() { return fence->GetCompletedValue(); }
    void   SetEventOnCompletion(UINT64 value, std::shared_ptr<EventWrapper> Event) { fence->SetEventOnCompletion(value, Event->Get()); }
};
class ID3D12PipelineStateWrapper {
public:
    ID3D12PipelineState *pipelineState = nullptr;

public:
    ID3D12PipelineStateWrapper(ID3D12PipelineState *pipelineState) : pipelineState(pipelineState) {}

    ~ID3D12PipelineStateWrapper() {
        if (pipelineState) pipelineState->Release();
    }

    std::string GetISA() {
        // https://github.com/baldurk/renderdoc/blob/2ace1fe84d62e2b2b5fdd25894dc3186864ee8fe/util/test/demos/dx/official/d3dcommon.h#L1041
#undef __DEFINE_GUID
#define __DEFINE_GUID(name, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) static const GUID name = { l, w1, w2, {b1, b2, b3, b4, b5, b6, b7, b8} }
        __DEFINE_GUID(WKPDID_CommentStringW, 0xd0149dc0, 0x90e8, 0x4ec8, 0x81, 0x44, 0xe9, 0x00, 0xad, 0x26, 0x6b, 0xb2);
#undef __DEFINE_GUID
        // https://github.com/baldurk/renderdoc/blob/ca4d79c0a34f4186e8a6ec39a0c9568b5c920ed8/renderdoc/driver/d3d12/d3d12_replay.cpp#L624
        UINT size = UINT(0);
        pipelineState->GetPrivateData(WKPDID_CommentStringW, &size, NULL);
        std::string isa = {};
        if (size != UINT(0)) {
            byte *data = new byte[size + UINT(1)];
            memset(data, UINT(0), size);
            pipelineState->GetPrivateData(WKPDID_CommentStringW, &size, data);
            char const *iter       = (char const *)data;
            char const *cdata_iter = strstr(iter, "![CDATA[");
            if (cdata_iter) {
                iter                     = cdata_iter + strlen("![CDATA[");
                char const *comment_iter = strstr(iter, "]]></comment>");
                if (comment_iter) {
                    isa = std::string(iter, size_t(comment_iter - iter));
                }
            } else {
                // Failed to find the right section
            }
            delete[] data;
        }
        return isa;
    }
};

class ID3D12RootSignatureWrapper {
public:
    ID3D12RootSignature *rootSignature = nullptr;

public:
    ID3D12RootSignatureWrapper(ID3D12RootSignature *rootSignature) : rootSignature(rootSignature) {}

    ~ID3D12RootSignatureWrapper() {
        if (rootSignature) rootSignature->Release();
    }
};

class ID3D12DescriptorHeapWrapper {
public:
    ID3D12DescriptorHeap *descriptorHeap = nullptr;

public:
    ID3D12DescriptorHeapWrapper(ID3D12DescriptorHeap *descriptorHeap) : descriptorHeap(descriptorHeap) {}

    ~ID3D12DescriptorHeapWrapper() {
        if (descriptorHeap) descriptorHeap->Release();
    }

    D3D12_CPU_DESCRIPTOR_HANDLE GetCPUDescriptorHandleForHeapStart() { return descriptorHeap->GetCPUDescriptorHandleForHeapStart(); }
    D3D12_GPU_DESCRIPTOR_HANDLE GetGPUDescriptorHandleForHeapStart() { return descriptorHeap->GetGPUDescriptorHandleForHeapStart(); }
};
class ID3D12CommandAllocatorWrapper {
public:
    ID3D12CommandAllocator *commandAllocator = nullptr;

public:
    ID3D12CommandAllocatorWrapper(ID3D12CommandAllocator *commandAllocator) : commandAllocator(commandAllocator) {}

    ~ID3D12CommandAllocatorWrapper() {
        if (commandAllocator) commandAllocator->Release();
    }

    void Reset() { commandAllocator->Reset(); }
};
class ID3D12CommandSignatureWrapper {
public:
    ID3D12CommandSignature *commandSignature = nullptr;

public:
    ID3D12CommandSignatureWrapper(ID3D12CommandSignature *commandSignature) : commandSignature(commandSignature) {}

    ~ID3D12CommandSignatureWrapper() {
        if (commandSignature) commandSignature->Release();
    }
};

class ID3D12ResourceWrapper {
public:
    ID3D12Resource * resource  = nullptr;
    ID3D12Resource1 *resource1 = nullptr;
    ID3D12Resource2 *resource2 = nullptr;

public:
    ID3D12ResourceWrapper(ID3D12Resource *resource) : resource(resource) {
        resource->QueryInterface(IID_PPV_ARGS(&resource1));
        resource->QueryInterface(IID_PPV_ARGS(&resource2));
    }
    ~ID3D12ResourceWrapper() {
        if (resource) resource->Release();
        if (resource1) resource1->Release();
        if (resource2) resource2->Release();
    }
    uint64_t GetGPUVirtualAddress() { return (uint64_t)resource->GetGPUVirtualAddress(); }
    uint64_t Map(UINT Subresource = 0, std::optional<D3D12_RANGE> ReadRange = std::nullopt) {
        void *data = nullptr;
        resource->Map(Subresource, ReadRange ? &ReadRange.value() : nullptr, &data);
        return (uint64_t)data;
    }
    void    Unmap(UINT Subresource = 0, std::optional<D3D12_RANGE> ReadRange = std::nullopt) { resource->Unmap(Subresource, ReadRange ? &ReadRange.value() : nullptr); }
    HRESULT WriteToSubresource(UINT DstSubresource, std::optional<D3D12_BOX> pDstBox, uint64_t pSrcData, UINT SrcRowPitch, UINT SrcDepthPitch) {
        return resource->WriteToSubresource(DstSubresource, pDstBox ? &pDstBox.value() : nullptr, (const void *)pSrcData, SrcRowPitch, SrcDepthPitch);
    }
    HRESULT ReadFromSubresource(uint64_t pDstData, UINT DstRowPitch, UINT DstDepthPitch, UINT SrcSubresource, std::optional<D3D12_BOX> pSrcBox) {
        return resource->ReadFromSubresource((void *)pDstData, DstRowPitch, DstDepthPitch, SrcSubresource, pSrcBox ? &pSrcBox.value() : nullptr);
    }
    std::pair<D3D12_HEAP_PROPERTIES, D3D12_HEAP_FLAGS> GetHeapProperties() {
        D3D12_HEAP_PROPERTIES heapProperties = {};
        D3D12_HEAP_FLAGS      heapFlags      = {};
        HRESULT               hr             = resource->GetHeapProperties(&heapProperties, &heapFlags);
        ASSERT_HRESULT_PANIC(hr);
        return {heapProperties, heapFlags};
    }
    D3D12_RESOURCE_DESC GetDesc() {
        D3D12_RESOURCE_DESC desc = resource->GetDesc();
        return desc;
    }
    // Resource1
    uint64_t GetProtectedResourceSession(GUID riid) {
        void *  protectedResourceSession = nullptr;
        HRESULT hr                       = resource1->GetProtectedResourceSession(riid, &protectedResourceSession);
        ASSERT_HRESULT_PANIC(hr);
        return (uint64_t)protectedResourceSession;
    }
    // Resource2
    D3D12_RESOURCE_DESC1 GetDesc1() {
        D3D12_RESOURCE_DESC1 desc = resource2->GetDesc1();
        return desc;
    }
    void SetName(std::wstring Name) { resource2->SetName(Name.c_str()); }
};

class D3D12_TEXTURE_COPY_LOCATION_WRAPPER {
public:
    std::shared_ptr<ID3D12ResourceWrapper> Resource = {};
    D3D12_TEXTURE_COPY_TYPE                Type     = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT     PlacedFootprint;
    UINT                                   SubresourceIndex;

public:
    D3D12_TEXTURE_COPY_LOCATION_WRAPPER(std::shared_ptr<ID3D12ResourceWrapper> Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT PlacedFootprint)
        : Resource(Resource), Type(D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT), PlacedFootprint(PlacedFootprint) {}
    D3D12_TEXTURE_COPY_LOCATION_WRAPPER(std::shared_ptr<ID3D12ResourceWrapper> Resource, UINT SubresourceIndex)
        : Resource(Resource), Type(D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX), SubresourceIndex(SubresourceIndex) {}

    D3D12_TEXTURE_COPY_LOCATION ToNative() {
        D3D12_TEXTURE_COPY_LOCATION location = {};
        location.pResource                   = Resource->resource;
        location.Type                        = Type;
        if (Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT) {
            location.PlacedFootprint = PlacedFootprint;
        } else {
            location.SubresourceIndex = SubresourceIndex;
        }
        return location;
    }
};

class D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_Wrapper {
public:
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE        Type          = {};
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS Flags         = {};
    UINT                                                NumDescs      = {};
    D3D12_ELEMENTS_LAYOUT                               DescsLayout   = {};
    D3D12_GPU_VIRTUAL_ADDRESS                           InstanceDescs = {};
    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC>         GeometryDescs = {};

public:
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_Wrapper(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS Flags, UINT NumDescs,
                                                                 D3D12_GPU_VIRTUAL_ADDRESS InstanceDescs) {
        Type                = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        this->Flags         = Flags;
        this->NumDescs      = NumDescs;
        DescsLayout         = D3D12_ELEMENTS_LAYOUT_ARRAY;
        this->InstanceDescs = InstanceDescs;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_Wrapper(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS Flags,
                                                                 std::vector<D3D12_RAYTRACING_GEOMETRY_DESC>         GeometryDescs) {
        Type                = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        this->Flags         = Flags;
        DescsLayout         = D3D12_ELEMENTS_LAYOUT_ARRAY;
        this->GeometryDescs = GeometryDescs;
    }
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_Wrapper() = default;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS ToNative() {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS desc = {};
        if (Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL) {
            desc.Type          = Type;
            desc.Flags         = Flags;
            desc.NumDescs      = NumDescs;
            desc.DescsLayout   = DescsLayout;
            desc.InstanceDescs = InstanceDescs;
        } else {
            desc.Type           = Type;
            desc.Flags          = Flags;
            desc.NumDescs       = (UINT)GeometryDescs.size();
            desc.DescsLayout    = DescsLayout;
            desc.pGeometryDescs = GeometryDescs.data();
        }
        return desc;
    }
};
class D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC_WRAPPER {
public:
    std::shared_ptr<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_Wrapper> Inputs                           = {};
    D3D12_GPU_VIRTUAL_ADDRESS                                                     SourceAccelerationStructureData  = {};
    D3D12_GPU_VIRTUAL_ADDRESS                                                     DestAccelerationStructureData    = {};
    D3D12_GPU_VIRTUAL_ADDRESS                                                     ScratchAccelerationStructureData = {};

public:
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC_WRAPPER(std::shared_ptr<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_Wrapper> Inputs,
                                                               D3D12_GPU_VIRTUAL_ADDRESS SourceAccelerationStructureData, D3D12_GPU_VIRTUAL_ADDRESS DestAccelerationStructureData,
                                                               D3D12_GPU_VIRTUAL_ADDRESS ScratchAccelerationStructureData) {
        this->Inputs                           = Inputs;
        this->SourceAccelerationStructureData  = SourceAccelerationStructureData;
        this->DestAccelerationStructureData    = DestAccelerationStructureData;
        this->ScratchAccelerationStructureData = ScratchAccelerationStructureData;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC ToNative() {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc = {};
        desc.Inputs                                             = Inputs->ToNative();
        desc.SourceAccelerationStructureData                    = SourceAccelerationStructureData;
        desc.DestAccelerationStructureData                      = DestAccelerationStructureData;
        desc.ScratchAccelerationStructureData                   = ScratchAccelerationStructureData;
        return desc;
    }
};

class ID3D12GraphicsCommandListWrapper {
public:
    ID3D12GraphicsCommandList * commandList  = nullptr;
    ID3D12GraphicsCommandList1 *commandList1 = nullptr;
    ID3D12GraphicsCommandList2 *commandList2 = nullptr;
    ID3D12GraphicsCommandList3 *commandList3 = nullptr;
    ID3D12GraphicsCommandList4 *commandList4 = nullptr;
    ID3D12GraphicsCommandList5 *commandList5 = nullptr;
    ID3D12GraphicsCommandList6 *commandList6 = nullptr;
    ID3D12GraphicsCommandList7 *commandList7 = nullptr;
    ID3D12GraphicsCommandList8 *commandList8 = nullptr;

public:
    ID3D12GraphicsCommandListWrapper(ID3D12GraphicsCommandList *commandList) : commandList(commandList) {
        commandList->QueryInterface(IID_PPV_ARGS(&commandList1));
        commandList->QueryInterface(IID_PPV_ARGS(&commandList2));
        commandList->QueryInterface(IID_PPV_ARGS(&commandList3));
        commandList->QueryInterface(IID_PPV_ARGS(&commandList4));
        commandList->QueryInterface(IID_PPV_ARGS(&commandList5));
        commandList->QueryInterface(IID_PPV_ARGS(&commandList6));
        commandList->QueryInterface(IID_PPV_ARGS(&commandList7));
        commandList->QueryInterface(IID_PPV_ARGS(&commandList8));
    }

    ~ID3D12GraphicsCommandListWrapper() {
        if (commandList) commandList->Release();
        if (commandList1) commandList1->Release();
        if (commandList2) commandList2->Release();
        if (commandList3) commandList3->Release();
        if (commandList4) commandList4->Release();
        if (commandList5) commandList5->Release();
        if (commandList6) commandList6->Release();
        if (commandList7) commandList7->Release();
        if (commandList8) commandList8->Release();
    }

    void Close() { commandList->Close(); }

    void Reset(std::shared_ptr<ID3D12CommandAllocatorWrapper> pAllocator, std::shared_ptr<ID3D12PipelineStateWrapper> pInitialState) {
        commandList->Reset(pAllocator->commandAllocator, pInitialState ? pInitialState->pipelineState : nullptr);
    }
    void ClearState(std::shared_ptr<ID3D12PipelineStateWrapper> pPipelineState) { commandList->ClearState(pPipelineState->pipelineState); }
    void DrawInstanced(UINT VertexCountPerInstance, UINT InstanceCount, UINT StartVertexLocation, UINT StartInstanceLocation) {
        commandList->DrawInstanced(VertexCountPerInstance, InstanceCount, StartVertexLocation, StartInstanceLocation);
    }
    void DrawIndexedInstanced(UINT IndexCountPerInstance, UINT InstanceCount, UINT StartIndexLocation, INT BaseVertexLocation, UINT StartInstanceLocation) {
        commandList->DrawIndexedInstanced(IndexCountPerInstance, InstanceCount, StartIndexLocation, BaseVertexLocation, StartInstanceLocation);
    }
    void SetPipelineState(std::shared_ptr<ID3D12PipelineStateWrapper> pPipelineState) { commandList->SetPipelineState(pPipelineState->pipelineState); }
    void SetGraphicsRootSignature(std::shared_ptr<ID3D12RootSignatureWrapper> pRootSignature) { commandList->SetGraphicsRootSignature(pRootSignature->rootSignature); }
    void SetDescriptorHeaps(std::vector<std::shared_ptr<ID3D12DescriptorHeapWrapper>> ppDescriptorHeaps) {
        UINT                  NumDescriptorHeaps = (UINT)ppDescriptorHeaps.size();
        ID3D12DescriptorHeap *descriptorHeaps[128];
        ASSERT_PANIC(NumDescriptorHeaps <= 128);
        for (UINT i = 0; i < NumDescriptorHeaps; i++) {
            descriptorHeaps[i] = ppDescriptorHeaps[i]->descriptorHeap;
        }
        commandList->SetDescriptorHeaps(NumDescriptorHeaps, descriptorHeaps);
    }
    void SetComputeRootSignature(std::shared_ptr<ID3D12RootSignatureWrapper> pRootSignature) { commandList->SetComputeRootSignature(pRootSignature->rootSignature); }
    void SetComputeRootDescriptorTable(UINT RootParameterIndex, D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor) {
        commandList->SetComputeRootDescriptorTable(RootParameterIndex, BaseDescriptor);
    }
    void BuildRaytracingAccelerationStructure(std::shared_ptr<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC_WRAPPER>             Desc,
                                              std::optional<std::vector<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC>> PostbuildInfoDescs) {
        UINT32 NumPostbuildInfoDescs = PostbuildInfoDescs ? (UINT32)PostbuildInfoDescs.value().size() : 0;
        ASSERT_PANIC(NumPostbuildInfoDescs <= 16);
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC postbuildInfoDescs[16];
        for (UINT32 i = 0; i < NumPostbuildInfoDescs; i++) {
            postbuildInfoDescs[i] = PostbuildInfoDescs ? PostbuildInfoDescs.value()[i] : D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC();
        }
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc = Desc->ToNative();
        commandList5->BuildRaytracingAccelerationStructure(&desc, NumPostbuildInfoDescs, postbuildInfoDescs);
    }
    void EmitRaytracingAccelerationStructurePostbuildInfo(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC Desc,
                                                          std::vector<D3D12_GPU_VIRTUAL_ADDRESS>                      SourceAccelerationStructureData) {
        UINT32 NumSourceAccelerationStructureData = (UINT32)SourceAccelerationStructureData.size();
        ASSERT_PANIC(NumSourceAccelerationStructureData <= 16);
        D3D12_GPU_VIRTUAL_ADDRESS sourceAccelerationStructureData[16];
        for (UINT32 i = 0; i < NumSourceAccelerationStructureData; i++) {
            sourceAccelerationStructureData[i] = SourceAccelerationStructureData[i];
        }
        commandList5->EmitRaytracingAccelerationStructurePostbuildInfo(&Desc, NumSourceAccelerationStructureData, sourceAccelerationStructureData);
    }
    void CopyRaytracingAccelerationStructure(D3D12_GPU_VIRTUAL_ADDRESS DestAccelerationStructureData, D3D12_GPU_VIRTUAL_ADDRESS SourceAccelerationStructureData,
                                             D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE Mode) {
        commandList5->CopyRaytracingAccelerationStructure(DestAccelerationStructureData, SourceAccelerationStructureData, Mode);
    }
    void SetGraphicsRootDescriptorTable(UINT RootParameterIndex, D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor) {
        commandList->SetGraphicsRootDescriptorTable(RootParameterIndex, BaseDescriptor);
    }
    void SetComputeRoot32BitConstant(UINT RootParameterIndex, UINT SrcData, UINT DestOffsetIn32BitValues) {
        commandList->SetComputeRoot32BitConstant(RootParameterIndex, SrcData, DestOffsetIn32BitValues);
    }
    void SetGraphicsRoot32BitConstant(UINT RootParameterIndex, UINT SrcData, UINT DestOffsetIn32BitValues) {
        commandList->SetGraphicsRoot32BitConstant(RootParameterIndex, SrcData, DestOffsetIn32BitValues);
    }
    void SetComputeRoot32BitConstants(UINT RootParameterIndex, UINT Num32BitValuesToSet, uint64_t pSrcData, UINT DestOffsetIn32BitValues) {
        commandList->SetComputeRoot32BitConstants(RootParameterIndex, Num32BitValuesToSet, (void *)pSrcData, DestOffsetIn32BitValues);
    }
    void SetGraphicsRoot32BitConstants(UINT RootParameterIndex, UINT Num32BitValuesToSet, uint64_t pSrcData, UINT DestOffsetIn32BitValues) {
        commandList->SetGraphicsRoot32BitConstants(RootParameterIndex, Num32BitValuesToSet, (void *)pSrcData, DestOffsetIn32BitValues);
    }
    void SetComputeRootConstantBufferView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation) {
        commandList->SetComputeRootConstantBufferView(RootParameterIndex, BufferLocation);
    }
    void SetGraphicsRootConstantBufferView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation) {
        commandList->SetGraphicsRootConstantBufferView(RootParameterIndex, BufferLocation);
    }
    void SetComputeRootShaderResourceView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation) {
        commandList->SetComputeRootShaderResourceView(RootParameterIndex, BufferLocation);
    }
    void SetGraphicsRootShaderResourceView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation) {
        commandList->SetGraphicsRootShaderResourceView(RootParameterIndex, BufferLocation);
    }
    void SetComputeRootUnorderedAccessView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation) {
        commandList->SetComputeRootUnorderedAccessView(RootParameterIndex, BufferLocation);
    }
    void SetGraphicsRootUnorderedAccessView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation) {
        commandList->SetGraphicsRootUnorderedAccessView(RootParameterIndex, BufferLocation);
    }
    void CopyBufferRegion(std::shared_ptr<ID3D12ResourceWrapper> pDstBuffer, UINT64 DstOffset, std::shared_ptr<ID3D12ResourceWrapper> pSrcBuffer, UINT64 SrcOffset,
                          UINT64 NumBytes) {
        commandList->CopyBufferRegion(pDstBuffer->resource, DstOffset, pSrcBuffer->resource, SrcOffset, NumBytes);
    }
    void CopyTextureRegion(std::shared_ptr<D3D12_TEXTURE_COPY_LOCATION_WRAPPER> Dst, UINT DstX, UINT DstY, UINT DstZ, std::shared_ptr<D3D12_TEXTURE_COPY_LOCATION_WRAPPER> Src,
                           std::optional<D3D12_BOX> SrcBox) {
        commandList->CopyTextureRegion(&Dst->ToNative(), DstX, DstY, DstZ, &Src->ToNative(), SrcBox ? &SrcBox.value() : nullptr);
    }
    void ResourceBarrier(std::vector<D3D12_RESOURCE_BARRIER> pBarriers) {
        UINT NumBarriers = (UINT)pBarriers.size();
        ASSERT_PANIC(NumBarriers <= 16);
        D3D12_RESOURCE_BARRIER barriers[16];
        for (UINT i = 0; i < NumBarriers; i++) {
            barriers[i] = pBarriers[i];
        }
        commandList->ResourceBarrier(NumBarriers, barriers);
    }
    void IASetPrimitiveTopology(D3D12_PRIMITIVE_TOPOLOGY PrimitiveTopology) { commandList->IASetPrimitiveTopology(PrimitiveTopology); }
    void IASetIndexBuffer(const D3D12_INDEX_BUFFER_VIEW &pView) { commandList->IASetIndexBuffer(&pView); }
    void IASetVertexBuffers(UINT StartSlot, std::vector<D3D12_VERTEX_BUFFER_VIEW> Views) {
        UINT NumViews = (UINT)Views.size();
        ASSERT_PANIC(NumViews <= 16);
        D3D12_VERTEX_BUFFER_VIEW views[16];
        for (UINT i = 0; i < NumViews; i++) {
            views[i] = Views[i];
        }
        commandList->IASetVertexBuffers(StartSlot, NumViews, views);
    }
    void RSSetViewports(std::vector<D3D12_VIEWPORT> Viewports) {
        UINT NumViewports = (UINT)Viewports.size();
        ASSERT_PANIC(NumViewports <= 16);
        D3D12_VIEWPORT viewports[16];
        for (UINT i = 0; i < NumViewports; i++) {
            viewports[i] = Viewports[i];
        }
        commandList->RSSetViewports(NumViewports, viewports);
    }
    void RSSetScissorRects(std::vector<D3D12_RECT> Rects) {
        UINT NumRects = (UINT)Rects.size();
        ASSERT_PANIC(NumRects <= 16);
        D3D12_RECT rects[16];
        for (UINT i = 0; i < NumRects; i++) {
            rects[i] = Rects[i];
        }
        commandList->RSSetScissorRects(NumRects, rects);
    }
    void OMSetRenderTargets(std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> RenderTargetDescriptors, BOOL RTsSingleHandleToDescriptorRange,
                            std::optional<D3D12_CPU_DESCRIPTOR_HANDLE> pDepthStencilDescriptor) {
        UINT NumRenderTargetDescriptors = (UINT)RenderTargetDescriptors.size();
        ASSERT_PANIC(NumRenderTargetDescriptors <= 8);
        D3D12_CPU_DESCRIPTOR_HANDLE renderTargetDescriptors[8];
        for (UINT i = 0; i < NumRenderTargetDescriptors; i++) {
            renderTargetDescriptors[i] = RenderTargetDescriptors[i];
        }
        commandList->OMSetRenderTargets(NumRenderTargetDescriptors, renderTargetDescriptors, RTsSingleHandleToDescriptorRange,
                                        pDepthStencilDescriptor ? &pDepthStencilDescriptor.value() : nullptr);
    }
    void ClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE RenderTargetView, std::vector<float> ColorRGBA, std::optional<std::vector<D3D12_RECT>> Rects) {
        UINT NumRects = Rects ? (UINT)Rects.value().size() : 0;
        ASSERT_PANIC(NumRects <= 8);
        ASSERT_PANIC(ColorRGBA.size() == 4);
        commandList->ClearRenderTargetView(RenderTargetView, ColorRGBA.data(), NumRects, Rects ? Rects.value().data() : nullptr);
    }
    void ClearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE DepthStencilView, D3D12_CLEAR_FLAGS ClearFlags, FLOAT Depth, UINT8 Stencil, std::vector<D3D12_RECT> Rects) {
        UINT NumRects = (UINT)Rects.size();
        commandList->ClearDepthStencilView(DepthStencilView, ClearFlags, Depth, Stencil, NumRects, Rects.data());
    }
    void Dispatch(UINT ThreadGroupCountX, UINT ThreadGroupCountY, UINT ThreadGroupCountZ) { commandList->Dispatch(ThreadGroupCountX, ThreadGroupCountY, ThreadGroupCountZ); }
    void ExecuteIndirect(std::shared_ptr<ID3D12CommandSignatureWrapper> pCommandSignature, UINT MaxCommandCount, std::shared_ptr<ID3D12ResourceWrapper> pArgumentBuffer,
                         UINT64 ArgumentBufferOffset, std::shared_ptr<ID3D12ResourceWrapper> pCountBuffer, UINT64 CountBufferOffset) {
        commandList->ExecuteIndirect(pCommandSignature->commandSignature, MaxCommandCount, pArgumentBuffer->resource, ArgumentBufferOffset, pCountBuffer->resource,
                                     CountBufferOffset);
    }
};

class ID3D12CommandQueueWrapper {
public:
    ID3D12CommandQueue *commandQueue = nullptr;

public:
    ID3D12CommandQueueWrapper(ID3D12CommandQueue *commandQueue) : commandQueue(commandQueue) {}

    ~ID3D12CommandQueueWrapper() {
        if (commandQueue) commandQueue->Release();
    }

    void Signal(std::shared_ptr<ID3D12FenceWrapper> fence, UINT64 value) { commandQueue->Signal(fence->fence, value); }
    void Wait(std::shared_ptr<ID3D12FenceWrapper> fence, UINT64 value) { commandQueue->Wait(fence->fence, value); }
    void ExecuteCommandLists(std::vector<std::shared_ptr<ID3D12GraphicsCommandListWrapper>> ppCommandLists) {
        UINT               NumCommandLists = (UINT)ppCommandLists.size();
        ID3D12CommandList *commandLists[128];
        ASSERT_PANIC(NumCommandLists <= 128);
        for (UINT i = 0; i < NumCommandLists; i++) {
            commandLists[i] = ppCommandLists[i]->commandList;
        }
        commandQueue->ExecuteCommandLists(NumCommandLists, commandLists);
    }

    // TODO
};

class IDXGISwapChainWrapper {
public:
    IDXGISwapChain * swapChain  = nullptr;
    IDXGISwapChain1 *swapChain1 = nullptr;
    IDXGISwapChain2 *swapChain2 = nullptr;
    IDXGISwapChain3 *swapChain3 = nullptr;
    IDXGISwapChain4 *swapChain4 = nullptr;

public:
    IDXGISwapChainWrapper(IDXGISwapChain *swapChain) : swapChain(swapChain) {
        swapChain->QueryInterface(IID_PPV_ARGS(&swapChain1));
        swapChain->QueryInterface(IID_PPV_ARGS(&swapChain2));
        swapChain->QueryInterface(IID_PPV_ARGS(&swapChain3));
        swapChain->QueryInterface(IID_PPV_ARGS(&swapChain4));
    }

    ~IDXGISwapChainWrapper() {
        if (swapChain) swapChain->Release();
        if (swapChain1) swapChain1->Release();
        if (swapChain2) swapChain2->Release();
        if (swapChain3) swapChain3->Release();
        if (swapChain4) swapChain4->Release();
    }

    HRESULT STDMETHODCALLTYPE Present(UINT SyncInterval, UINT Flags) { return swapChain->Present(SyncInterval, Flags); }

    std::shared_ptr<ID3D12ResourceWrapper> GetBuffer(UINT Buffer) {
        ID3D12Resource *resource = nullptr;
        HRESULT         hr       = swapChain->GetBuffer(Buffer, IID_PPV_ARGS(&resource));
        ASSERT_HRESULT_PANIC(hr);
        return std::make_shared<ID3D12ResourceWrapper>(resource);
    }

    uint32_t GetCurrentBackBufferIndex() { return swapChain3->GetCurrentBackBufferIndex(); }

    void ResizeBuffers(UINT BufferCount, UINT Width, UINT Height, DXGI_FORMAT NewFormat, UINT SwapChainFlags) {
        swapChain->ResizeBuffers(BufferCount, Width, Height, NewFormat, SwapChainFlags);
    }
};

class IDXGIAdapterWrapper {
public:
    IDXGIAdapter * adapter  = nullptr;
    IDXGIAdapter1 *adapter1 = nullptr;
    IDXGIAdapter2 *adapter2 = nullptr;
    IDXGIAdapter3 *adapter3 = nullptr;
    IDXGIAdapter4 *adapter4 = nullptr;

public:
    IDXGIAdapterWrapper(IDXGIAdapter *adapter) : adapter(adapter) {
        adapter->QueryInterface(IID_PPV_ARGS(&adapter1));
        adapter->QueryInterface(IID_PPV_ARGS(&adapter2));
        adapter->QueryInterface(IID_PPV_ARGS(&adapter3));
        adapter->QueryInterface(IID_PPV_ARGS(&adapter4));
    }

    std::shared_ptr<DXGI_ADAPTER_DESC_WRAPPER> GetDesc() {
        DXGI_ADAPTER_DESC desc = {};
        HRESULT           hr   = adapter->GetDesc(&desc);
        ASSERT_HRESULT_PANIC(hr);
        return std::make_shared<DXGI_ADAPTER_DESC_WRAPPER>(desc);
    }

    ~IDXGIAdapterWrapper() {
        if (adapter) adapter->Release();
        if (adapter1) adapter1->Release();
        if (adapter2) adapter2->Release();
        if (adapter3) adapter3->Release();
        if (adapter4) adapter4->Release();
    }

    DXGI_ADAPTER_DESC3 GetDesc3() {
        DXGI_ADAPTER_DESC3 desc;
        adapter4->GetDesc3(&desc);
        return desc;
    }
};

class ID3D12DebugWrapper {
public:
    ID3D12Debug * debug  = nullptr;
    ID3D12Debug1 *debug1 = nullptr;

public:
    ID3D12DebugWrapper() {
        HRESULT hr = D3D12GetDebugInterface(IID_PPV_ARGS(&debug));
        ASSERT_HRESULT_PANIC(hr);
        debug->QueryInterface(IID_PPV_ARGS(&debug1));
    }

    ~ID3D12DebugWrapper() {
        if (debug) debug->Release();
        if (debug1) debug1->Release();
    }

    void EnableDebugLayer() { debug->EnableDebugLayer(); }

    void SetEnableGPUBasedValidation(BOOL Enable) { debug1->SetEnableGPUBasedValidation(Enable); }
};

#if 0
class DXGI_SWAP_CHAIN_DESC_Wrapper {
public:
    DXGI_MODE_DESC   BufferDesc;
    DXGI_SAMPLE_DESC SampleDesc;
    DXGI_USAGE       BufferUsage;
    UINT             BufferCount;
    uint64_t         OutputWindow;
    BOOL             Windowed;
    DXGI_SWAP_EFFECT SwapEffect;
    UINT             Flags;

public:
    DXGI_SWAP_CHAIN_DESC_Wrapper(DXGI_MODE_DESC BufferDesc, DXGI_SAMPLE_DESC SampleDesc, DXGI_USAGE BufferUsage, UINT BufferCount, uint64_t OutputWindow, BOOL Windowed,
                                 DXGI_SWAP_EFFECT SwapEffect, UINT Flags)
        : BufferDesc(BufferDesc), SampleDesc(SampleDesc), BufferUsage(BufferUsage), BufferCount(BufferCount), OutputWindow(OutputWindow), Windowed(Windowed),
          SwapEffect(SwapEffect), Flags(Flags) {}

    DXGI_SWAP_CHAIN_DESC ToNative() {
        DXGI_SWAP_CHAIN_DESC desc = {};
        desc.BufferDesc           = BufferDesc;
        desc.SampleDesc           = SampleDesc;
        desc.BufferUsage          = BufferUsage;
        desc.BufferCount          = BufferCount;
        desc.OutputWindow         = (HWND)OutputWindow;
        desc.Windowed             = Windowed;
        desc.SwapEffect           = SwapEffect;
        desc.Flags                = Flags;
        return desc;
    }
};
#endif

class IDXGIFactoryWrapper {
public:
    IDXGIFactory * factory  = nullptr;
    IDXGIFactory1 *factory1 = nullptr;
    IDXGIFactory2 *factory2 = nullptr;
    IDXGIFactory3 *factory3 = nullptr;
    IDXGIFactory4 *factory4 = nullptr;
    IDXGIFactory5 *factory5 = nullptr;
    IDXGIFactory6 *factory6 = nullptr;
    IDXGIFactory7 *factory7 = nullptr;

public:
    IDXGIFactoryWrapper() {
        HRESULT hr = CreateDXGIFactory(IID_PPV_ARGS(&factory));
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to create IDXGIFactory");
        }

        factory->QueryInterface(IID_PPV_ARGS(&factory1));
        factory->QueryInterface(IID_PPV_ARGS(&factory2));
        factory->QueryInterface(IID_PPV_ARGS(&factory3));
        factory->QueryInterface(IID_PPV_ARGS(&factory4));
        factory->QueryInterface(IID_PPV_ARGS(&factory5));
        factory->QueryInterface(IID_PPV_ARGS(&factory6));
        factory->QueryInterface(IID_PPV_ARGS(&factory7));
    }

    ~IDXGIFactoryWrapper() {
        if (factory) factory->Release();
        if (factory1) factory1->Release();
        if (factory2) factory2->Release();
        if (factory3) factory3->Release();
        if (factory4) factory4->Release();
        if (factory5) factory5->Release();
        if (factory6) factory6->Release();
        if (factory7) factory7->Release();
    }

    std::shared_ptr<IDXGIAdapterWrapper> GetAdapter(UINT index) {
        IDXGIAdapter *adapter = nullptr;
        factory->EnumAdapters(index, &adapter);
        return std::make_shared<IDXGIAdapterWrapper>(adapter);
    }

    std::vector<std::shared_ptr<IDXGIAdapterWrapper>> EnumAdapters() {
        std::vector<std::shared_ptr<IDXGIAdapterWrapper>> adapters = {};
        for (UINT i = 0;; i++) {
            IDXGIAdapter *adapter = nullptr;
            if (factory->EnumAdapters(i, &adapter) == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            ASSERT_PANIC(adapter);
            adapters.push_back(std::make_shared<IDXGIAdapterWrapper>(adapter));
        }
        return adapters;
    }

    // virtual HRESULT STDMETHODCALLTYPE CreateSwapChain(
    //     /* [annotation][in] */
    //     _In_ IUnknown *pDevice,
    //     /* [annotation][in] */
    //     _In_ DXGI_SWAP_CHAIN_DESC *pDesc,
    //     /* [annotation][out] */
    //     _COM_Outptr_ IDXGISwapChain **ppSwapChain) = 0;
    std::shared_ptr<IDXGISwapChainWrapper> CreateSwapChain(std::shared_ptr<ID3D12CommandQueueWrapper> commandQueue, DXGI_SWAP_CHAIN_DESC desc) {
        IDXGISwapChain *swapChain = nullptr;
        HRESULT         hr        = factory->CreateSwapChain(commandQueue->commandQueue, &desc, &swapChain);
        ASSERT_HRESULT_PANIC(hr);
        return std::make_shared<IDXGISwapChainWrapper>(swapChain);
    }
};

class D3D12_SHADER_BYTECODE_Wrapper {
public:
    std::vector<uint8_t> Bytecode;

public:
    D3D12_SHADER_BYTECODE_Wrapper(std::vector<uint8_t> Bytecode) : Bytecode(Bytecode) {}

    D3D12_SHADER_BYTECODE ToNative() {
        D3D12_SHADER_BYTECODE bytecode = {};
        bytecode.pShaderBytecode       = Bytecode.data();
        bytecode.BytecodeLength        = Bytecode.size();
        return bytecode;
    }
};

class D3D12_INPUT_ELEMENT_DESC_Wrapper {
public:
    std::string                SemanticName         = {};
    UINT                       SemanticIndex        = {};
    DXGI_FORMAT                Format               = {};
    UINT                       InputSlot            = {};
    UINT                       AlignedByteOffset    = {};
    D3D12_INPUT_CLASSIFICATION InputSlotClass       = {};
    UINT                       InstanceDataStepRate = {};

public:
    D3D12_INPUT_ELEMENT_DESC_Wrapper() = default;
    D3D12_INPUT_ELEMENT_DESC_Wrapper(std::string SemanticName, UINT SemanticIndex, DXGI_FORMAT Format, UINT InputSlot, UINT AlignedByteOffset,
                                     D3D12_INPUT_CLASSIFICATION InputSlotClass, UINT InstanceDataStepRate)
        : SemanticName(SemanticName), SemanticIndex(SemanticIndex), Format(Format), InputSlot(InputSlot), AlignedByteOffset(AlignedByteOffset), InputSlotClass(InputSlotClass),
          InstanceDataStepRate(InstanceDataStepRate) {}

    D3D12_INPUT_ELEMENT_DESC ToNative() {
        D3D12_INPUT_ELEMENT_DESC desc = {};
        desc.SemanticName             = SemanticName.c_str();
        desc.SemanticIndex            = SemanticIndex;
        desc.Format                   = Format;
        desc.InputSlot                = InputSlot;
        desc.AlignedByteOffset        = AlignedByteOffset;
        desc.InputSlotClass           = InputSlotClass;
        desc.InstanceDataStepRate     = InstanceDataStepRate;
        return desc;
    }
};

class D3D12_GRAPHICS_PIPELINE_STATE_DESC_Wrapper {
public:
    std::shared_ptr<ID3D12RootSignatureWrapper>                    pRootSignature        = {};
    std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper>                 VS                    = {};
    std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper>                 PS                    = {};
    std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper>                 DS                    = {};
    std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper>                 HS                    = {};
    std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper>                 GS                    = {};
    D3D12_STREAM_OUTPUT_DESC                                       StreamOutput          = {};
    D3D12_BLEND_DESC                                               BlendState            = {};
    UINT                                                           SampleMask            = {};
    D3D12_RASTERIZER_DESC                                          RasterizerState       = {};
    D3D12_DEPTH_STENCIL_DESC                                       DepthStencilState     = {};
    std::vector<std::shared_ptr<D3D12_INPUT_ELEMENT_DESC_Wrapper>> InputLayouts          = {};
    D3D12_INDEX_BUFFER_STRIP_CUT_VALUE                             IBStripCutValue       = {};
    D3D12_PRIMITIVE_TOPOLOGY_TYPE                                  PrimitiveTopologyType = {};
    UINT                                                           NumRenderTargets      = {};
    std::vector<DXGI_FORMAT>                                       RTVFormats            = {};
    DXGI_FORMAT                                                    DSVFormat             = {};
    DXGI_SAMPLE_DESC                                               SampleDesc            = {};
    UINT                                                           NodeMask              = {};
    std::optional<D3D12_CACHED_PIPELINE_STATE>                     CachedPSO             = {};
    D3D12_PIPELINE_STATE_FLAGS                                     Flags                 = {};

    std::vector<D3D12_INPUT_ELEMENT_DESC> cache_InputLayouts = {};

public:
    D3D12_GRAPHICS_PIPELINE_STATE_DESC_Wrapper(std::shared_ptr<ID3D12RootSignatureWrapper>                    pRootSignature    = {}, //
                                               std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper>                 VS                = {}, //
                                               std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper>                 PS                = {}, //
                                               std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper>                 DS                = {}, //
                                               std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper>                 HS                = {}, //
                                               std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper>                 GS                = {}, //
                                               D3D12_STREAM_OUTPUT_DESC                                       StreamOutput      = {}, //
                                               D3D12_BLEND_DESC                                               BlendState        = {}, //
                                               UINT                                                           SampleMask        = {}, //
                                               D3D12_RASTERIZER_DESC                                          RasterizerState   = {}, //
                                               D3D12_DEPTH_STENCIL_DESC                                       DepthStencilState = {}, //
                                               std::vector<std::shared_ptr<D3D12_INPUT_ELEMENT_DESC_Wrapper>> InputLayouts      = {}, //
                                               D3D12_INDEX_BUFFER_STRIP_CUT_VALUE IBStripCutValue = {}, D3D12_PRIMITIVE_TOPOLOGY_TYPE PrimitiveTopologyType = {},
                                               UINT NumRenderTargets = {}, std::vector<DXGI_FORMAT> RTVFormats = {}, DXGI_FORMAT DSVFormat = {}, DXGI_SAMPLE_DESC SampleDesc = {},
                                               UINT NodeMask = {}, std::optional<D3D12_CACHED_PIPELINE_STATE> CachedPSO = {}, D3D12_PIPELINE_STATE_FLAGS Flags = {})
        : pRootSignature(pRootSignature), VS(VS), PS(PS), DS(DS), HS(HS), GS(GS), StreamOutput(StreamOutput), BlendState(BlendState), SampleMask(SampleMask),
          RasterizerState(RasterizerState), DepthStencilState(DepthStencilState), InputLayouts(InputLayouts), IBStripCutValue(IBStripCutValue),
          PrimitiveTopologyType(PrimitiveTopologyType), NumRenderTargets(NumRenderTargets), RTVFormats(RTVFormats), DSVFormat(DSVFormat), SampleDesc(SampleDesc),
          NodeMask(NodeMask), CachedPSO(CachedPSO), Flags(Flags) {}

    D3D12_GRAPHICS_PIPELINE_STATE_DESC ToNative() {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature                     = pRootSignature->rootSignature;
        desc.VS                                 = VS ? VS->ToNative() : D3D12_SHADER_BYTECODE();
        desc.PS                                 = PS ? PS->ToNative() : D3D12_SHADER_BYTECODE();
        desc.DS                                 = DS ? DS->ToNative() : D3D12_SHADER_BYTECODE();
        desc.HS                                 = HS ? HS->ToNative() : D3D12_SHADER_BYTECODE();
        desc.GS                                 = GS ? GS->ToNative() : D3D12_SHADER_BYTECODE();
        desc.StreamOutput                       = StreamOutput;
        desc.BlendState                         = BlendState;
        desc.SampleMask                         = SampleMask;
        desc.RasterizerState                    = RasterizerState;
        desc.DepthStencilState                  = DepthStencilState;
        desc.InputLayout.NumElements            = (UINT)InputLayouts.size();
        cache_InputLayouts.resize(desc.InputLayout.NumElements);
        for (UINT i = 0; i < desc.InputLayout.NumElements; i++) {
            cache_InputLayouts[i] = InputLayouts[i]->ToNative();
        }
        desc.InputLayout.pInputElementDescs = cache_InputLayouts.data();
        desc.IBStripCutValue                = IBStripCutValue;
        desc.PrimitiveTopologyType          = PrimitiveTopologyType;
        desc.NumRenderTargets               = NumRenderTargets;
        for (UINT i = 0; i < NumRenderTargets; i++) {
            desc.RTVFormats[i] = RTVFormats[i];
        }
        for (UINT i = NumRenderTargets; i < 8; i++) {
            desc.RTVFormats[i] = DXGI_FORMAT_UNKNOWN;
        }
        desc.DSVFormat  = DSVFormat;
        desc.SampleDesc = SampleDesc;
        desc.NodeMask   = NodeMask;
        desc.CachedPSO  = CachedPSO ? CachedPSO.value() : D3D12_CACHED_PIPELINE_STATE{};
        desc.Flags      = Flags;
        return desc;
    }
};
class D3D12_COMPUTE_PIPELINE_STATE_DESC_Wrapper {
    std::shared_ptr<ID3D12RootSignatureWrapper>    pRootSignature = {};
    std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper> CS             = {};
    UINT                                           NodeMask       = {};
    std::optional<D3D12_CACHED_PIPELINE_STATE>     CachedPSO      = {};
    D3D12_PIPELINE_STATE_FLAGS                     Flags          = {};

public:
    D3D12_COMPUTE_PIPELINE_STATE_DESC_Wrapper(std::shared_ptr<ID3D12RootSignatureWrapper>    pRootSignature = {}, //
                                              std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper> CS             = {}, //
                                              UINT                                           NodeMask       = {}, //
                                              std::optional<D3D12_CACHED_PIPELINE_STATE>     CachedPSO      = {}, //
                                              D3D12_PIPELINE_STATE_FLAGS                     Flags          = {})
        : pRootSignature(pRootSignature), CS(CS), NodeMask(NodeMask), CachedPSO(CachedPSO), Flags(Flags) {}

    D3D12_COMPUTE_PIPELINE_STATE_DESC ToNative() {
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature                    = pRootSignature->rootSignature;
        desc.CS                                = CS ? CS->ToNative() : D3D12_SHADER_BYTECODE();
        desc.NodeMask                          = NodeMask;
        desc.CachedPSO                         = CachedPSO ? CachedPSO.value() : D3D12_CACHED_PIPELINE_STATE{};
        desc.Flags                             = Flags;
        return desc;
    }
};

class CopyableFootprints {
public:
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts;
    std::vector<UINT>                               NumRows;
    std::vector<UINT64>                             RowSizeInBytes;
    std::vector<UINT64>                             TotalBytes;
};

class ID3D12DeviceWrapper {
public:
    ID3D12Device * device  = nullptr;
    ID3D12Device1 *device1 = nullptr;
    ID3D12Device2 *device2 = nullptr;
    ID3D12Device3 *device3 = nullptr;
    ID3D12Device4 *device4 = nullptr;
    ID3D12Device5 *device5 = nullptr;
    ID3D12Device6 *device6 = nullptr;
    ID3D12Device7 *device7 = nullptr;
    ID3D12Device8 *device8 = nullptr;

public:
    ID3D12DeviceWrapper(ID3D12Device *device) : device(device) {
        device->QueryInterface(IID_PPV_ARGS(&device1));
        device->QueryInterface(IID_PPV_ARGS(&device2));
        device->QueryInterface(IID_PPV_ARGS(&device3));
        device->QueryInterface(IID_PPV_ARGS(&device4));
        device->QueryInterface(IID_PPV_ARGS(&device5));
        device->QueryInterface(IID_PPV_ARGS(&device6));
        device->QueryInterface(IID_PPV_ARGS(&device7));
        device->QueryInterface(IID_PPV_ARGS(&device8));
    }

    ~ID3D12DeviceWrapper() {
        if (device) device->Release();
        if (device1) device1->Release();
        if (device2) device2->Release();
        if (device3) device3->Release();
        if (device4) device4->Release();
        if (device5) device5->Release();
        if (device6) device6->Release();
        if (device7) device7->Release();
        if (device8) device8->Release();
    }

    std::shared_ptr<CopyableFootprints> GetCopyableFootprints(const D3D12_RESOURCE_DESC &resourceDesc, UINT firstSubresource, UINT NumSubresources, UINT64 BaseOffset) {
        std::shared_ptr<CopyableFootprints> footprints = std::make_shared<CopyableFootprints>();
        footprints->layouts.resize(NumSubresources);
        footprints->NumRows.resize(NumSubresources);
        footprints->RowSizeInBytes.resize(NumSubresources);
        footprints->TotalBytes.resize(NumSubresources);
        device->GetCopyableFootprints(&resourceDesc, firstSubresource, NumSubresources, BaseOffset, footprints->layouts.data(), footprints->NumRows.data(),
                                      footprints->RowSizeInBytes.data(), footprints->TotalBytes.data());
        return footprints;
    }

    std::shared_ptr<ID3D12ResourceWrapper> CreateCommittedResource(const D3D12_HEAP_PROPERTIES &heapProperties, D3D12_HEAP_FLAGS heapFlags, const D3D12_RESOURCE_DESC &resourceDesc,
                                                                   D3D12_RESOURCE_STATES initialState, const std::optional<D3D12_CLEAR_VALUE> &optimizedClearValue) {
        ID3D12Resource *resource = nullptr;
        HRESULT         hr = device->CreateCommittedResource(&heapProperties, heapFlags, &resourceDesc, initialState, optimizedClearValue ? &optimizedClearValue.value() : nullptr,
                                                     IID_PPV_ARGS(&resource));
        ASSERT_HRESULT_PANIC(hr);
        return std::make_shared<ID3D12ResourceWrapper>(resource);
    }
    std::shared_ptr<ID3D12CommandAllocatorWrapper> CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE type) {
        ID3D12CommandAllocator *commandAllocator = nullptr;
        HRESULT                 hr               = device->CreateCommandAllocator(type, IID_PPV_ARGS(&commandAllocator));
        ASSERT_HRESULT_PANIC(hr);
        return std::make_shared<ID3D12CommandAllocatorWrapper>(commandAllocator);
    }
    std::shared_ptr<ID3D12GraphicsCommandListWrapper> CreateCommandList(UINT nodeMask, D3D12_COMMAND_LIST_TYPE type,
                                                                        std::shared_ptr<ID3D12CommandAllocatorWrapper> pCommandAllocator,
                                                                        std::shared_ptr<ID3D12PipelineStateWrapper>    pInitialState) {
        ID3D12GraphicsCommandList *commandList = nullptr;
        HRESULT                    hr =
            device->CreateCommandList(nodeMask, type, pCommandAllocator->commandAllocator, pInitialState ? pInitialState->pipelineState : nullptr, IID_PPV_ARGS(&commandList));
        ASSERT_HRESULT_PANIC(hr);
        return std::make_shared<ID3D12GraphicsCommandListWrapper>(commandList);
    }
    std::shared_ptr<ID3D12DescriptorHeapWrapper> CreateDescriptorHeap(const D3D12_DESCRIPTOR_HEAP_DESC &descriptorHeapDesc) {
        ID3D12DescriptorHeap *descriptorHeap = nullptr;
        HRESULT               hr             = device->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(&descriptorHeap));
        ASSERT_HRESULT_PANIC(hr);
        return std::make_shared<ID3D12DescriptorHeapWrapper>(descriptorHeap);
    }
    std::shared_ptr<ID3D12RootSignatureWrapper> CreateRootSignature(UINT nodeMask, py::bytes Bytes) {
        std::string          _bytes        = Bytes;
        ID3D12RootSignature *rootSignature = nullptr;
        HRESULT              hr            = device->CreateRootSignature(nodeMask, (void *)_bytes.data(), _bytes.size(), IID_PPV_ARGS(&rootSignature));
        ASSERT_HRESULT_PANIC(hr);
        return std::make_shared<ID3D12RootSignatureWrapper>(rootSignature);
    }
    std::shared_ptr<ID3D12PipelineStateWrapper> CreateGraphicsPipelineState(std::shared_ptr<D3D12_GRAPHICS_PIPELINE_STATE_DESC_Wrapper> desc) {
        ID3D12PipelineState *pipelineState = nullptr;
        HRESULT              hr            = device->CreateGraphicsPipelineState(&desc->ToNative(), IID_PPV_ARGS(&pipelineState));
        ASSERT_HRESULT_PANIC(hr);
        return std::make_shared<ID3D12PipelineStateWrapper>(pipelineState);
    }
    UINT GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE DescriptorHeapType) { return device->GetDescriptorHandleIncrementSize(DescriptorHeapType); }
    std::shared_ptr<ID3D12PipelineStateWrapper> CreateComputePipelineState(std::shared_ptr<D3D12_COMPUTE_PIPELINE_STATE_DESC_Wrapper> desc) {
        ID3D12PipelineState *pipelineState = nullptr;
        HRESULT              hr            = device->CreateComputePipelineState(&desc->ToNative(), IID_PPV_ARGS(&pipelineState));
        ASSERT_HRESULT_PANIC(hr);
        return std::make_shared<ID3D12PipelineStateWrapper>(pipelineState);
    }
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO
    GetRaytracingAccelerationStructurePrebuildInfo(std::shared_ptr<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_Wrapper> Inputs) {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info = {};
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS  desc = Inputs->ToNative();
        device5->GetRaytracingAccelerationStructurePrebuildInfo(&desc, &info);
        return info;
    }
    std::shared_ptr<ID3D12CommandQueueWrapper> CreateCommandQueue(const D3D12_COMMAND_QUEUE_DESC &desc) {
        ID3D12CommandQueue *commandQueue = nullptr;
        HRESULT             hr           = device->CreateCommandQueue(&desc, IID_PPV_ARGS(&commandQueue));
        ASSERT_HRESULT_PANIC(hr);
        return std::make_shared<ID3D12CommandQueueWrapper>(commandQueue);
    }
    std::shared_ptr<ID3D12FenceWrapper> CreateFence(UINT64 initialValue, D3D12_FENCE_FLAGS flags) {
        ID3D12Fence *fence = nullptr;
        HRESULT      hr    = device->CreateFence(initialValue, flags, IID_PPV_ARGS(&fence));
        ASSERT_HRESULT_PANIC(hr);
        return std::make_shared<ID3D12FenceWrapper>(fence);
    }
    void CreateRenderTargetView(std::shared_ptr<ID3D12ResourceWrapper> pResource, D3D12_RENDER_TARGET_VIEW_DESC Desc, D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor) {
        device->CreateRenderTargetView(pResource->resource, &Desc, DestDescriptor);
    }
    void CreateDepthStencilView(std::shared_ptr<ID3D12ResourceWrapper> pResource, D3D12_DEPTH_STENCIL_VIEW_DESC Desc, D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor) {
        device->CreateDepthStencilView(pResource->resource, &Desc, DestDescriptor);
    }
    void CreateConstantBufferView(const D3D12_CONSTANT_BUFFER_VIEW_DESC &Desc, D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor) {
        device->CreateConstantBufferView(&Desc, DestDescriptor);
    }
    void CreateShaderResourceView(std::shared_ptr<ID3D12ResourceWrapper> pResource, D3D12_SHADER_RESOURCE_VIEW_DESC Desc, D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor) {
        device->CreateShaderResourceView(pResource->resource, &Desc, DestDescriptor);
    }
    void CreateUnorderedAccessView(std::shared_ptr<ID3D12ResourceWrapper> pResource, std::shared_ptr<ID3D12ResourceWrapper> pCounterResource, D3D12_UNORDERED_ACCESS_VIEW_DESC Desc,
                                   D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor) {
        device->CreateUnorderedAccessView(pResource->resource, pCounterResource ? pCounterResource->resource : nullptr, &Desc, DestDescriptor);
    }
    void CreateSampler(const D3D12_SAMPLER_DESC &Desc, D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor) { device->CreateSampler(&Desc, DestDescriptor); }
};

static std::shared_ptr<ID3D12DeviceWrapper> CreateDevice(std::shared_ptr<IDXGIAdapterWrapper> adapter, D3D_FEATURE_LEVEL featureLevel) {
    ID3D12Device *device = nullptr;
    HRESULT       hr     = D3D12CreateDevice(adapter->adapter, featureLevel, IID_PPV_ARGS(&device));
    ASSERT_HRESULT_PANIC(hr);
    return std::make_shared<ID3D12DeviceWrapper>(device);
}

static py::tuple GetWindowSize(uint64_t hwnd_int) {
    HWND hwnd = reinterpret_cast<HWND>(hwnd_int);
    RECT rect;
    if (GetClientRect(hwnd, &rect)) {
        int width  = rect.right - rect.left;
        int height = rect.bottom - rect.top;
        return py::make_tuple(width, height);
    }
    throw std::runtime_error("Unable to retrieve window size.");
}

static bool _IsDebuggerPresent() { return ::IsDebuggerPresent(); }

class ID3D12SDKConfigurationWrapper {
public:
    ID3D12SDKConfiguration *sdkConfiguration = nullptr;

public:
    ID3D12SDKConfigurationWrapper() {
#undef __DEFINE_GUID
#define __DEFINE_GUID(name, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) static const GUID name = { l, w1, w2, {b1, b2, b3, b4, b5, b6, b7, b8} }
        __DEFINE_GUID(CLSID_D3D12SDKConfiguration, 0x7cda6aca, 0xa03e, 0x49c8, 0x94, 0x58, 0x03, 0x34, 0xd2, 0x0e, 0x07, 0xce);
#undef __DEFINE_GUID

        HMODULE d3d12 = GetModuleHandleA("d3d12.dll");
        ASSERT_PANIC(NULL != d3d12);
        auto D3D12GetInterfacePfn = (PFN_D3D12_GET_INTERFACE)GetProcAddress(d3d12, "D3D12GetInterface");
        ASSERT_PANIC(NULL != D3D12GetInterfacePfn);
        HRESULT hr = D3D12GetInterfacePfn(CLSID_D3D12SDKConfiguration, IID_PPV_ARGS(&sdkConfiguration));
        ASSERT_HRESULT_PANIC(hr);
    }
    ~ID3D12SDKConfigurationWrapper() {
        if (sdkConfiguration) sdkConfiguration->Release();
    }
    void SetSDKVersion(UINT SDKVersion, std::string SDKPath) {
        HRESULT hr = sdkConfiguration->SetSDKVersion(SDKVersion, SDKPath.c_str());
        ASSERT_HRESULT_PANIC(hr);
    }
};

void export_d3d12_0(py::module &m) {
    m.def("square", &square);
    m.def("GetWindowSize", &GetWindowSize);
    m.def("IsDebuggerPresent", &_IsDebuggerPresent);

    py::class_<ID3D12SDKConfigurationWrapper, std::shared_ptr<ID3D12SDKConfigurationWrapper>>(m, "ID3D12SDKConfiguration")
        .def(py::init<>())
        .def("SetSDKVersion", &ID3D12SDKConfigurationWrapper::SetSDKVersion);

    py::enum_<DXGI_SWAP_EFFECT>(m, "DXGI_SWAP_EFFECT")
        .value("DISCARD", DXGI_SWAP_EFFECT_DISCARD)
        .value("SEQUENTIAL", DXGI_SWAP_EFFECT_SEQUENTIAL)
        .value("FLIP_SEQUENTIAL", DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL)
        .value("FLIP_DISCARD", DXGI_SWAP_EFFECT_FLIP_DISCARD)
        .export_values();

    py::enum_<DXGI_FORMAT>(m, "DXGI_FORMAT")
        .value("UNKNOWN", DXGI_FORMAT_UNKNOWN)
        .value("R32G32B32A32_TYPELESS", DXGI_FORMAT_R32G32B32A32_TYPELESS)
        .value("R32G32B32A32_FLOAT", DXGI_FORMAT_R32G32B32A32_FLOAT)
        .value("R32G32B32A32_UINT", DXGI_FORMAT_R32G32B32A32_UINT)
        .value("R32G32B32A32_SINT", DXGI_FORMAT_R32G32B32A32_SINT)
        .value("R32G32B32_TYPELESS", DXGI_FORMAT_R32G32B32_TYPELESS)
        .value("R32G32B32_FLOAT", DXGI_FORMAT_R32G32B32_FLOAT)
        .value("R32G32B32_UINT", DXGI_FORMAT_R32G32B32_UINT)
        .value("R32G32B32_SINT", DXGI_FORMAT_R32G32B32_SINT)
        .value("R16G16B16A16_TYPELESS", DXGI_FORMAT_R16G16B16A16_TYPELESS)
        .value("R16G16B16A16_FLOAT", DXGI_FORMAT_R16G16B16A16_FLOAT)
        .value("R16G16B16A16_UNORM", DXGI_FORMAT_R16G16B16A16_UNORM)
        .value("R16G16B16A16_UINT", DXGI_FORMAT_R16G16B16A16_UINT)
        .value("R16G16B16A16_SNORM", DXGI_FORMAT_R16G16B16A16_SNORM)
        .value("R16G16B16A16_SINT", DXGI_FORMAT_R16G16B16A16_SINT)
        .value("R32G32_TYPELESS", DXGI_FORMAT_R32G32_TYPELESS)
        .value("R32G32_FLOAT", DXGI_FORMAT_R32G32_FLOAT)
        .value("R32G32_UINT", DXGI_FORMAT_R32G32_UINT)
        .value("R32G32_SINT", DXGI_FORMAT_R32G32_SINT)
        .value("R32G8X24_TYPELESS", DXGI_FORMAT_R32G8X24_TYPELESS)
        .value("D32_FLOAT_S8X24_UINT", DXGI_FORMAT_D32_FLOAT_S8X24_UINT)
        .value("R32_FLOAT_X8X24_TYPELESS", DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS)
        .value("X32_TYPELESS_G8X24_UINT", DXGI_FORMAT_X32_TYPELESS_G8X24_UINT)
        .value("R10G10B10A2_TYPELESS", DXGI_FORMAT_R10G10B10A2_TYPELESS)
        .value("R10G10B10A2_UNORM", DXGI_FORMAT_R10G10B10A2_UNORM)
        .value("R10G10B10A2_UINT", DXGI_FORMAT_R10G10B10A2_UINT)
        .value("R11G11B10_FLOAT", DXGI_FORMAT_R11G11B10_FLOAT)
        .value("R8G8B8A8_TYPELESS", DXGI_FORMAT_R8G8B8A8_TYPELESS)
        .value("R8G8B8A8_UNORM", DXGI_FORMAT_R8G8B8A8_UNORM)
        .value("R8G8B8A8_UNORM_SRGB", DXGI_FORMAT_R8G8B8A8_UNORM_SRGB)
        .value("R8G8B8A8_UINT", DXGI_FORMAT_R8G8B8A8_UINT)
        .value("R8G8B8A8_SNORM", DXGI_FORMAT_R8G8B8A8_SNORM)
        .value("R8G8B8A8_SINT", DXGI_FORMAT_R8G8B8A8_SINT)
        .value("R16G16_TYPELESS", DXGI_FORMAT_R16G16_TYPELESS)
        .value("R16G16_FLOAT", DXGI_FORMAT_R16G16_FLOAT)
        .value("R16G16_UNORM", DXGI_FORMAT_R16G16_UNORM)
        .value("R16G16_UINT", DXGI_FORMAT_R16G16_UINT)
        .value("R16G16_SNORM", DXGI_FORMAT_R16G16_SNORM)
        .value("R16G16_SINT", DXGI_FORMAT_R16G16_SINT)
        .value("R32_TYPELESS", DXGI_FORMAT_R32_TYPELESS)
        .value("D32_FLOAT", DXGI_FORMAT_D32_FLOAT)
        .value("R32_FLOAT", DXGI_FORMAT_R32_FLOAT)
        .value("R32_UINT", DXGI_FORMAT_R32_UINT)
        .value("R32_SINT", DXGI_FORMAT_R32_SINT)
        .value("R24G8_TYPELESS", DXGI_FORMAT_R24G8_TYPELESS)
        .value("D24_UNORM_S8_UINT", DXGI_FORMAT_D24_UNORM_S8_UINT)
        .value("R24_UNORM_X8_TYPELESS", DXGI_FORMAT_R24_UNORM_X8_TYPELESS)
        .value("X24_TYPELESS_G8_UINT", DXGI_FORMAT_X24_TYPELESS_G8_UINT)
        .value("R8G8_TYPELESS", DXGI_FORMAT_R8G8_TYPELESS)
        .value("R8G8_UNORM", DXGI_FORMAT_R8G8_UNORM)
        .value("R8G8_UINT", DXGI_FORMAT_R8G8_UINT)
        .value("R8G8_SNORM", DXGI_FORMAT_R8G8_SNORM)
        .value("R8G8_SINT", DXGI_FORMAT_R8G8_SINT)
        .value("R16_TYPELESS", DXGI_FORMAT_R16_TYPELESS)
        .value("R16_FLOAT", DXGI_FORMAT_R16_FLOAT)
        .value("D16_UNORM", DXGI_FORMAT_D16_UNORM)
        .value("R16_UNORM", DXGI_FORMAT_R16_UNORM)
        .value("R16_UINT", DXGI_FORMAT_R16_UINT)
        .value("R16_SNORM", DXGI_FORMAT_R16_SNORM)
        .value("R16_SINT", DXGI_FORMAT_R16_SINT)
        .value("R8_TYPELESS", DXGI_FORMAT_R8_TYPELESS)
        .value("R8_UNORM", DXGI_FORMAT_R8_UNORM)
        .value("R8_UINT", DXGI_FORMAT_R8_UINT)
        .value("R8_SNORM", DXGI_FORMAT_R8_SNORM)
        .value("R8_SINT", DXGI_FORMAT_R8_SINT)
        .value("A8_UNORM", DXGI_FORMAT_A8_UNORM)
        .value("R1_UNORM", DXGI_FORMAT_R1_UNORM)
        .value("R9G9B9E5_SHAREDEXP", DXGI_FORMAT_R9G9B9E5_SHAREDEXP)
        .value("R8G8_B8G8_UNORM", DXGI_FORMAT_R8G8_B8G8_UNORM)
        .value("G8R8_G8B8_UNORM", DXGI_FORMAT_G8R8_G8B8_UNORM)
        .value("BC1_TYPELESS", DXGI_FORMAT_BC1_TYPELESS)
        .value("BC1_UNORM", DXGI_FORMAT_BC1_UNORM)
        .value("BC1_UNORM_SRGB", DXGI_FORMAT_BC1_UNORM_SRGB)
        .value("BC2_TYPELESS", DXGI_FORMAT_BC2_TYPELESS)
        .value("BC2_UNORM", DXGI_FORMAT_BC2_UNORM)
        .value("BC2_UNORM_SRGB", DXGI_FORMAT_BC2_UNORM_SRGB)
        .value("BC3_TYPELESS", DXGI_FORMAT_BC3_TYPELESS)
        .value("BC3_UNORM", DXGI_FORMAT_BC3_UNORM)
        .value("BC3_UNORM_SRGB", DXGI_FORMAT_BC3_UNORM_SRGB)
        .value("BC4_TYPELESS", DXGI_FORMAT_BC4_TYPELESS)
        .value("BC4_UNORM", DXGI_FORMAT_BC4_UNORM)
        .value("BC4_SNORM", DXGI_FORMAT_BC4_SNORM)
        .value("BC5_TYPELESS", DXGI_FORMAT_BC5_TYPELESS)
        .value("BC5_UNORM", DXGI_FORMAT_BC5_UNORM)
        .value("BC5_SNORM", DXGI_FORMAT_BC5_SNORM)
        .value("B5G6R5_UNORM", DXGI_FORMAT_B5G6R5_UNORM)
        .value("B5G5R5A1_UNORM", DXGI_FORMAT_B5G5R5A1_UNORM)
        .value("B8G8R8A8_UNORM", DXGI_FORMAT_B8G8R8A8_UNORM)
        .value("B8G8R8X8_UNORM", DXGI_FORMAT_B8G8R8X8_UNORM)
        .value("R10G10B10_XR_BIAS_A2_UNORM", DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM)
        .value("B8G8R8A8_TYPELESS", DXGI_FORMAT_B8G8R8A8_TYPELESS)
        .value("B8G8R8A8_UNORM_SRGB", DXGI_FORMAT_B8G8R8A8_UNORM_SRGB)
        .value("B8G8R8X8_TYPELESS", DXGI_FORMAT_B8G8R8X8_TYPELESS)
        .value("B8G8R8X8_UNORM_SRGB", DXGI_FORMAT_B8G8R8X8_UNORM_SRGB)
        .value("BC6H_TYPELESS", DXGI_FORMAT_BC6H_TYPELESS)
        .value("BC6H_UF16", DXGI_FORMAT_BC6H_UF16)
        .value("BC6H_SF16", DXGI_FORMAT_BC6H_SF16)
        .value("BC7_TYPELESS", DXGI_FORMAT_BC7_TYPELESS)
        .value("BC7_UNORM", DXGI_FORMAT_BC7_UNORM)
        .value("BC7_UNORM_SRGB", DXGI_FORMAT_BC7_UNORM_SRGB)
        .value("AYUV", DXGI_FORMAT_AYUV)
        .value("Y410", DXGI_FORMAT_Y410)
        .value("Y416", DXGI_FORMAT_Y416)
        .value("NV12", DXGI_FORMAT_NV12)
        .value("P010", DXGI_FORMAT_P010)
        .value("P016", DXGI_FORMAT_P016)
        .value("420_OPAQUE", DXGI_FORMAT_420_OPAQUE)
        .value("YUY2", DXGI_FORMAT_YUY2)
        .value("Y210", DXGI_FORMAT_Y210)
        .value("Y216", DXGI_FORMAT_Y216)
        .value("NV11", DXGI_FORMAT_NV11)
        .value("AI44", DXGI_FORMAT_AI44)
        .value("IA44", DXGI_FORMAT_IA44)
        .value("P8", DXGI_FORMAT_P8)
        .value("A8P8", DXGI_FORMAT_A8P8)
        .value("B4G4R4A4_UNORM", DXGI_FORMAT_B4G4R4A4_UNORM)
        .export_values();

    py::class_<IDXGIAdapterWrapper, std::shared_ptr<IDXGIAdapterWrapper>>(m, "IDXGIAdapter") //
        .def(py::init<IDXGIAdapter *>())                                                     //
        .def("GetDesc", &IDXGIAdapterWrapper::GetDesc)                                       //
        ;

    py::class_<IDXGISwapChainWrapper, std::shared_ptr<IDXGISwapChainWrapper>>(m, "IDXGISwapChain")                                              //
        .def(py::init<IDXGISwapChain *>())                                                                                                      //
        .def("GetBuffer", &IDXGISwapChainWrapper::GetBuffer)                                                                                    //
        .def("Present", &IDXGISwapChainWrapper::Present)                                                                                        //
        .def("ResizeBuffers", &IDXGISwapChainWrapper::ResizeBuffers, "BufferCount"_a, "Width"_a, "Height"_a, "NewFormat"_a, "SwapChainFlags"_a) //
        .def("GetCurrentBackBufferIndex", &IDXGISwapChainWrapper::GetCurrentBackBufferIndex)                                                    //

        ;

    py::enum_<DXGI_MODE_SCALING>(m, "DXGI_MODE_SCALING")     //
        .value("UNSPECIFIED", DXGI_MODE_SCALING_UNSPECIFIED) //
        .value("CENTERED", DXGI_MODE_SCALING_CENTERED)       //
        .value("STRETCHED", DXGI_MODE_SCALING_STRETCHED)     //
        .export_values();
    py::enum_<DXGI_MODE_SCANLINE_ORDER>(m, "DXGI_MODE_SCANLINE_ORDER")          //
        .value("UNSPECIFIED", DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED)             //
        .value("PROGRESSIVE", DXGI_MODE_SCANLINE_ORDER_PROGRESSIVE)             //
        .value("UPPER_FIELD_FIRST", DXGI_MODE_SCANLINE_ORDER_UPPER_FIELD_FIRST) //
        .value("LOWER_FIELD_FIRST", DXGI_MODE_SCANLINE_ORDER_LOWER_FIELD_FIRST) //
        .export_values();
    py::class_<DXGI_SAMPLE_DESC>(m, "DXGI_SAMPLE_DESC") //
        .def(py::init([](UINT Count, UINT Quality) {
                 DXGI_SAMPLE_DESC desc = {};
                 desc.Count            = Count;
                 desc.Quality          = Quality;
                 return desc;
             }),
             "Count"_a, "Quality"_a)                          //
        .def_readwrite("Count", &DXGI_SAMPLE_DESC::Count)     //
        .def_readwrite("Quality", &DXGI_SAMPLE_DESC::Quality) //
        ;

    py::class_<DXGI_RATIONAL>(m, "DXGI_RATIONAL") //
        .def(py::init([](UINT Numerator, UINT Denominator) {
                 DXGI_RATIONAL rational = {};
                 rational.Numerator     = Numerator;
                 rational.Denominator   = Denominator;
                 return rational;
             }),
             "Numerator"_a, "Denominator"_a)                       //
        .def_readwrite("Numerator", &DXGI_RATIONAL::Numerator)     //
        .def_readwrite("Denominator", &DXGI_RATIONAL::Denominator) //
        ;
    py::class_<DXGI_MODE_DESC>(m, "DXGI_MODE_DESC") //
        .def(py::init([](UINT width, UINT height, DXGI_RATIONAL refreshRate, DXGI_FORMAT format, DXGI_MODE_SCANLINE_ORDER scanlineOrdering, DXGI_MODE_SCALING scaling) {
                 DXGI_MODE_DESC desc   = {};
                 desc.Width            = width;
                 desc.Height           = height;
                 desc.RefreshRate      = refreshRate;
                 desc.Format           = format;
                 desc.ScanlineOrdering = scanlineOrdering;
                 desc.Scaling          = scaling;
                 return desc;
             }),
             "Width"_a, "Height"_a, "RefreshRate"_a, "Format"_a, "ScanlineOrdering"_a, "Scaling"_a) //
        .def_readwrite("Width", &DXGI_MODE_DESC::Width)                                             //
        .def_readwrite("Height", &DXGI_MODE_DESC::Height)                                           //
        .def_readwrite("RefreshRate", &DXGI_MODE_DESC::RefreshRate)                                 //
        .def_readwrite("Format", &DXGI_MODE_DESC::Format)                                           //
        .def_readwrite("ScanlineOrdering", &DXGI_MODE_DESC::ScanlineOrdering)                       //
        .def_readwrite("Scaling", &DXGI_MODE_DESC::Scaling)                                         //
        ;

    enum DXGI_USAGE_Proxy : uint32_t {
        NONE                 = 0,
        BACK_BUFFER          = DXGI_USAGE_BACK_BUFFER,
        SHADER_INPUT         = DXGI_USAGE_SHADER_INPUT,
        RENDER_TARGET_OUTPUT = DXGI_USAGE_RENDER_TARGET_OUTPUT,
        SHARED               = DXGI_USAGE_SHARED,
        READ_ONLY            = DXGI_USAGE_READ_ONLY,
        DISCARD_ON_PRESENT   = DXGI_USAGE_DISCARD_ON_PRESENT,
        UNORDERED_ACCESS     = DXGI_USAGE_UNORDERED_ACCESS,
    };

    py::enum_<DXGI_USAGE_Proxy>(m, "DXGI_USAGE", py::arithmetic())             //
        .value("BACK_BUFFER", DXGI_USAGE_Proxy::BACK_BUFFER)                   //
        .value("SHADER_INPUT", DXGI_USAGE_Proxy::SHADER_INPUT)                 //
        .value("RENDER_TARGET_OUTPUT", DXGI_USAGE_Proxy::RENDER_TARGET_OUTPUT) //
        .value("SHARED", DXGI_USAGE_Proxy::SHARED)                             //
        .value("READ_ONLY", DXGI_USAGE_Proxy::READ_ONLY)                       //
        .value("DISCARD_ON_PRESENT", DXGI_USAGE_Proxy::DISCARD_ON_PRESENT)     //
        .value("UNORDERED_ACCESS", DXGI_USAGE_Proxy::UNORDERED_ACCESS)         //
        .export_values();

    py::enum_<DXGI_SWAP_CHAIN_FLAG>(m, "DXGI_SWAP_CHAIN_FLAG", py::arithmetic())                        //
        .value("NONPREROTATED", DXGI_SWAP_CHAIN_FLAG_NONPREROTATED)                                     //
        .value("ALLOW_MODE_SWITCH", DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH)                             //
        .value("GDI_COMPATIBLE", DXGI_SWAP_CHAIN_FLAG_GDI_COMPATIBLE)                                   //
        .value("RESTRICTED_CONTENT", DXGI_SWAP_CHAIN_FLAG_RESTRICTED_CONTENT)                           //
        .value("RESTRICT_SHARED_RESOURCE_DRIVER", DXGI_SWAP_CHAIN_FLAG_RESTRICT_SHARED_RESOURCE_DRIVER) //
        .value("DISPLAY_ONLY", DXGI_SWAP_CHAIN_FLAG_DISPLAY_ONLY)                                       //
        .value("FRAME_LATENCY_WAITABLE_OBJECT", DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT)     //
        .value("FOREGROUND_LAYER", DXGI_SWAP_CHAIN_FLAG_FOREGROUND_LAYER)                               //
        .value("FULLSCREEN_VIDEO", DXGI_SWAP_CHAIN_FLAG_FULLSCREEN_VIDEO)                               //
        .value("YUV_VIDEO", DXGI_SWAP_CHAIN_FLAG_YUV_VIDEO)                                             //
        .export_values();

    py::class_<DXGI_SWAP_CHAIN_DESC>(m, "DXGI_SWAP_CHAIN_DESC") //
        .def(py::init())                                        //
        .def(py::init([](DXGI_MODE_DESC BufferDesc, DXGI_SAMPLE_DESC SampleDesc, uint32_t BufferUsage, UINT BufferCount, uint64_t OutputWindow, BOOL Windowed,
                         DXGI_SWAP_EFFECT SwapEffect, int Flags = (int)0) {
                 return DXGI_SWAP_CHAIN_DESC{BufferDesc, SampleDesc, (uint32_t)BufferUsage, BufferCount, (HWND)OutputWindow, Windowed, SwapEffect, (UINT)Flags};
             }),
             "BufferDesc"_a, "SampleDesc"_a, "BufferUsage"_a, "BufferCount"_a, "OutputWindow"_a, "Windowed"_a, "SwapEffect"_a, "Flags"_a) //
        ;

    // py::class_<DXGI_SWAP_CHAIN_DESC_Wrapper>(m, "DXGI_SWAP_CHAIN_DESC") //
    //     .def(py::init())                                                //
    //     .def(py::init([](DXGI_MODE_DESC BufferDesc, DXGI_SAMPLE_DESC SampleDesc, DXGI_USAGE BufferUsage, UINT BufferCount, uint64_t OutputWindow, BOOL Windowed,
    //                      DXGI_SWAP_EFFECT SwapEffect,
    //                      UINT Flags) { return DXGI_SWAP_CHAIN_DESC_Wrapper{BufferDesc, SampleDesc, BufferUsage, BufferCount, OutputWindow, Windowed, SwapEffect, Flags}; }), //
    //          "BufferDesc"_a, "SampleDesc"_a, "BufferUsage"_a, "BufferCount"_a, "OutputWindow"_a, "Windowed"_a, "SwapEffect"_a, "Flags"_a)                                    //
    //     .def_readwrite("BufferDesc", &DXGI_SWAP_CHAIN_DESC_Wrapper::BufferDesc)                                                                                              //
    //     .def_readwrite("SampleDesc", &DXGI_SWAP_CHAIN_DESC_Wrapper::SampleDesc)                                                                                              //
    //     .def_readwrite("BufferUsage", &DXGI_SWAP_CHAIN_DESC_Wrapper::BufferUsage)                                                                                            //
    //     .def_readwrite("BufferCount", &DXGI_SWAP_CHAIN_DESC_Wrapper::BufferCount)                                                                                            //
    //     .def_readwrite("OutputWindow", &DXGI_SWAP_CHAIN_DESC_Wrapper::OutputWindow)                                                                                          //
    //     .def_readwrite("Windowed", &DXGI_SWAP_CHAIN_DESC_Wrapper::Windowed)                                                                                                  //
    //     .def_readwrite("SwapEffect", &DXGI_SWAP_CHAIN_DESC_Wrapper::SwapEffect)                                                                                              //
    //     .def_readwrite("Flags", &DXGI_SWAP_CHAIN_DESC_Wrapper::Flags)                                                                                                        //
    //     ;

    py::class_<IDXGIFactoryWrapper, std::shared_ptr<IDXGIFactoryWrapper>>(m, "IDXGIFactory") //
        .def(py::init())                                                                     //
        .def("GetAdapter", &IDXGIFactoryWrapper::GetAdapter)                                 //
        .def("EnumAdapters", &IDXGIFactoryWrapper::EnumAdapters)                             //
        .def("CreateSwapChain", &IDXGIFactoryWrapper::CreateSwapChain)                       //
        ;

    py::class_<ID3D12RootSignatureWrapper, std::shared_ptr<ID3D12RootSignatureWrapper>>(m, "ID3D12RootSignature") //
        ;

    py::class_<DXGI_ADAPTER_DESC_WRAPPER, std::shared_ptr<DXGI_ADAPTER_DESC_WRAPPER>>(m, "DXGI_ADAPTER_DESC") //
        .def(py::init<DXGI_ADAPTER_DESC>())                                                                   //
        .def_readonly("Description", &DXGI_ADAPTER_DESC_WRAPPER::Description)                                 //
        .def_readonly("VendorId", &DXGI_ADAPTER_DESC_WRAPPER::VendorId)                                       //
        .def_readonly("DeviceId", &DXGI_ADAPTER_DESC_WRAPPER::DeviceId)                                       //
        .def_readonly("SubSysId", &DXGI_ADAPTER_DESC_WRAPPER::SubSysId)                                       //
        .def_readonly("Revision", &DXGI_ADAPTER_DESC_WRAPPER::Revision)                                       //
        .def_readonly("DedicatedVideoMemory", &DXGI_ADAPTER_DESC_WRAPPER::DedicatedVideoMemory)               //
        .def_readonly("DedicatedSystemMemory", &DXGI_ADAPTER_DESC_WRAPPER::DedicatedSystemMemory)             //
        .def_readonly("SharedSystemMemory", &DXGI_ADAPTER_DESC_WRAPPER::SharedSystemMemory)                   //
        .def_readonly("AdapterLuid", &DXGI_ADAPTER_DESC_WRAPPER::AdapterLuid)                                 //
        ;
    py::class_<LUID>(m, "LUID").def(py::init()).def_readwrite("LowPart", &LUID::LowPart).def_readwrite("HighPart", &LUID::HighPart);

    py::enum_<D3D12_PIPELINE_STATE_FLAGS>(m, "D3D12_PIPELINE_STATE_FLAGS", py::arithmetic())
        .value("NONE", D3D12_PIPELINE_STATE_FLAG_NONE)
        .value("TOOL_DEBUG", D3D12_PIPELINE_STATE_FLAG_TOOL_DEBUG)
        .value("DYNAMIC_DEPTH_BIAS", D3D12_PIPELINE_STATE_FLAG_DYNAMIC_DEPTH_BIAS)
        .value("DYNAMIC_INDEX_BUFFER_STRIP_CUT", D3D12_PIPELINE_STATE_FLAG_DYNAMIC_INDEX_BUFFER_STRIP_CUT)
        .export_values();

    py::enum_<D3D_FEATURE_LEVEL>(m, "D3D_FEATURE_LEVEL")
        .value("_1_0_CORE", D3D_FEATURE_LEVEL_1_0_CORE)
        .value("_9_1", D3D_FEATURE_LEVEL_9_1)
        .value("_9_2", D3D_FEATURE_LEVEL_9_2)
        .value("_9_3", D3D_FEATURE_LEVEL_9_3)
        .value("_10_0", D3D_FEATURE_LEVEL_10_0)
        .value("_10_1", D3D_FEATURE_LEVEL_10_1)
        .value("_11_0", D3D_FEATURE_LEVEL_11_0)
        .value("_11_1", D3D_FEATURE_LEVEL_11_1)
        .value("_12_0", D3D_FEATURE_LEVEL_12_0)
        .value("_12_1", D3D_FEATURE_LEVEL_12_1)
        .export_values();

    py::class_<D3D12_SHADER_BYTECODE_Wrapper, std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper>>(m, "D3D12_SHADER_BYTECODE") //
        .def(py::init([](py::bytes bytecode) {                                                                            //
            std::vector<uint8_t> bytecodeVec = {};
            bytecodeVec.resize(py::len(bytecode));
            std::string bytecodeStr = bytecode;
            std::memcpy(bytecodeVec.data(), bytecodeStr.data(), bytecodeVec.size());
            return D3D12_SHADER_BYTECODE_Wrapper(bytecodeVec);
        })) //
        ;
    py::class_<D3D12_COMPUTE_PIPELINE_STATE_DESC_Wrapper, std::shared_ptr<D3D12_COMPUTE_PIPELINE_STATE_DESC_Wrapper>>(m, "D3D12_COMPUTE_PIPELINE_STATE_DESC") //
        .def(py::init([](std::shared_ptr<ID3D12RootSignatureWrapper>    pRootSignature,                                                                       //
                         std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper> CS,                                                                                   //
                         UINT                                           NodeMask,                                                                             //
                         std::optional<D3D12_CACHED_PIPELINE_STATE>     CachedPSO,                                                                            //
                         D3D12_PIPELINE_STATE_FLAGS                     Flags) {                                                                                                  //
                 return D3D12_COMPUTE_PIPELINE_STATE_DESC_Wrapper(pRootSignature, CS, NodeMask, CachedPSO, Flags);
             }),                                                                                                                                         //
             "RootSignature"_a = nullptr, "CS"_a = nullptr, "NodeMask"_a = 0, "CachedPSO"_a = std::nullopt, "Flags"_a = D3D12_PIPELINE_STATE_FLAG_NONE) //
        ;

    py::class_<D3D12_DEPTH_STENCIL_VALUE>(m, "D3D12_DEPTH_STENCIL_VALUE") //
        .def(py::init())                                                  //
        .def_readwrite("Depth", &D3D12_DEPTH_STENCIL_VALUE::Depth)        //
        .def_readwrite("Stencil", &D3D12_DEPTH_STENCIL_VALUE::Stencil)    //
        ;
    py::class_<D3D12_CLEAR_VALUE>(m, "D3D12_CLEAR_VALUE") //
        .def(py::init([](DXGI_FORMAT format, float color[4]) {
                 D3D12_CLEAR_VALUE value = {};
                 value.Format            = format;
                 value.Color[0]          = color[0];
                 value.Color[1]          = color[1];
                 value.Color[2]          = color[2];
                 value.Color[3]          = color[3];
                 return value;
             }),                                             //
             "Format"_a, "Color"_a)                          //
        .def_readwrite("Format", &D3D12_CLEAR_VALUE::Format) //
        .def_property(
            "Color", [](D3D12_CLEAR_VALUE &self) { return self.Color; },
            [](D3D12_CLEAR_VALUE &self, float color[4]) {
                self.Color[0] = color[0];
                self.Color[1] = color[1];
                self.Color[2] = color[2];
                self.Color[3] = color[3];
            }) //
        .def_readwrite("DepthStencil", &D3D12_CLEAR_VALUE::DepthStencil);

    py::enum_<D3D12_RESOURCE_STATES>(m, "D3D12_RESOURCE_STATES", py::arithmetic())
        .value("COMMON", D3D12_RESOURCE_STATE_COMMON)
        .value("VERTEX_AND_CONSTANT_BUFFER", D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER)
        .value("INDEX_BUFFER", D3D12_RESOURCE_STATE_INDEX_BUFFER)
        .value("RENDER_TARGET", D3D12_RESOURCE_STATE_RENDER_TARGET)
        .value("UNORDERED_ACCESS", D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        .value("DEPTH_WRITE", D3D12_RESOURCE_STATE_DEPTH_WRITE)
        .value("DEPTH_READ", D3D12_RESOURCE_STATE_DEPTH_READ)
        .value("NON_PIXEL_SHADER_RESOURCE", D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        .value("PIXEL_SHADER_RESOURCE", D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
        .value("STREAM_OUT", D3D12_RESOURCE_STATE_STREAM_OUT)
        .value("INDIRECT_ARGUMENT", D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)
        .value("COPY_DEST", D3D12_RESOURCE_STATE_COPY_DEST)
        .value("COPY_SOURCE", D3D12_RESOURCE_STATE_COPY_SOURCE)
        .value("RESOLVE_DEST", D3D12_RESOURCE_STATE_RESOLVE_DEST)
        .value("RESOLVE_SOURCE", D3D12_RESOURCE_STATE_RESOLVE_SOURCE)
        .value("GENERIC_READ", D3D12_RESOURCE_STATE_GENERIC_READ)
        .value("PRESENT", D3D12_RESOURCE_STATE_PRESENT)
        .value("PREDICATION", D3D12_RESOURCE_STATE_PREDICATION)
        .value("VIDEO_DECODE_READ", D3D12_RESOURCE_STATE_VIDEO_DECODE_READ)
        .value("VIDEO_DECODE_WRITE", D3D12_RESOURCE_STATE_VIDEO_DECODE_WRITE)
        .value("VIDEO_PROCESS_READ", D3D12_RESOURCE_STATE_VIDEO_PROCESS_READ)
        .value("VIDEO_PROCESS_WRITE", D3D12_RESOURCE_STATE_VIDEO_PROCESS_WRITE)
        .value("VIDEO_ENCODE_READ", D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ)
        .value("VIDEO_ENCODE_WRITE", D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE)
        .value("RAYTRACING_ACCELERATION_STRUCTURE", D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE)
        .export_values();

    py::enum_<D3D12_HEAP_FLAGS>(m, "D3D12_HEAP_FLAGS")
        .value("NONE", D3D12_HEAP_FLAG_NONE)
        .value("SHARED", D3D12_HEAP_FLAG_SHARED)
        .value("DENY_BUFFERS", D3D12_HEAP_FLAG_DENY_BUFFERS)
        .value("DENY_TEXTURES", D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES)
        .value("DENY_RT_DS_TEXTURES", D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES)
        .value("HARDWARE_PROTECTED", D3D12_HEAP_FLAG_HARDWARE_PROTECTED)
        .value("ALLOW_WRITE_WATCH", D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH)
        .value("ALLOW_SHADER_ATOMICS", D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS)
        .value("CREATE_NOT_RESIDENT", D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT)
        .value("CREATE_NOT_ZEROED", D3D12_HEAP_FLAG_CREATE_NOT_ZEROED)
        .value("TOOLS_USE_MANUAL_WRITE_TRACKING", D3D12_HEAP_FLAG_TOOLS_USE_MANUAL_WRITE_TRACKING)
        .value("ALLOW_ALL_BUFFERS_AND_TEXTURES", D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES)
        .value("ALLOW_ONLY_BUFFERS", D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS)
        .value("ALLOW_ONLY_NON_RT_DS_TEXTURES", D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES)
        .value("ALLOW_ONLY_RT_DS_TEXTURES", D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES)
        .export_values();

    py::enum_<D3D12_MEMORY_POOL>(m, "D3D12_MEMORY_POOL")
        .value("UNKNOWN", D3D12_MEMORY_POOL_UNKNOWN)
        .value("L0", D3D12_MEMORY_POOL_L0)
        .value("L1", D3D12_MEMORY_POOL_L1)
        .export_values();

    py::enum_<D3D12_CPU_PAGE_PROPERTY>(m, "D3D12_CPU_PAGE_PROPERTY")
        .value("UNKNOWN", D3D12_CPU_PAGE_PROPERTY_UNKNOWN)
        .value("NOT_AVAILABLE", D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE)
        .value("WRITE_COMBINE", D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE)
        .value("WRITE_BACK", D3D12_CPU_PAGE_PROPERTY_WRITE_BACK)
        .export_values();

    py::enum_<D3D12_HEAP_TYPE>(m, "D3D12_HEAP_TYPE")
        .value("DEFAULT", D3D12_HEAP_TYPE_DEFAULT)
        .value("UPLOAD", D3D12_HEAP_TYPE_UPLOAD)
        .value("READBACK", D3D12_HEAP_TYPE_READBACK)
        .value("CUSTOM", D3D12_HEAP_TYPE_CUSTOM)
        .export_values();

    py::class_<D3D12_HEAP_PROPERTIES>(m, "D3D12_HEAP_PROPERTIES") //
        .def(py::init())                                          //
        .def(py::init([](D3D12_HEAP_TYPE Type, D3D12_CPU_PAGE_PROPERTY CPUPageProperty, D3D12_MEMORY_POOL MemoryPoolPreference, UINT CreationNodeMask, UINT VisibleNodeMask) {
                 D3D12_HEAP_PROPERTIES properties = {};
                 properties.Type                  = Type;
                 properties.CPUPageProperty       = CPUPageProperty;
                 properties.MemoryPoolPreference  = MemoryPoolPreference;
                 properties.CreationNodeMask      = CreationNodeMask;
                 properties.VisibleNodeMask       = VisibleNodeMask;
                 return properties;
             }),
             "Type"_a            = D3D12_HEAP_TYPE_DEFAULT,                                                                                                                  //
             "CPUPageProperty"_a = D3D12_CPU_PAGE_PROPERTY_UNKNOWN, "MemoryPoolPreference"_a = D3D12_MEMORY_POOL_UNKNOWN, "CreationNodeMask"_a = 1, "VisibleNodeMask"_a = 1) //

        .def_readwrite("Type", &D3D12_HEAP_PROPERTIES::Type)                                 //
        .def_readwrite("CPUPageProperty", &D3D12_HEAP_PROPERTIES::CPUPageProperty)           //
        .def_readwrite("MemoryPoolPreference", &D3D12_HEAP_PROPERTIES::MemoryPoolPreference) //
        .def_readwrite("CreationNodeMask", &D3D12_HEAP_PROPERTIES::CreationNodeMask)         //
        .def_readwrite("VisibleNodeMask", &D3D12_HEAP_PROPERTIES::VisibleNodeMask)           //
        ;

    py::class_<GUID>(m, "GUID")
        .def_readwrite("Data1)", &GUID::Data1)
        .def_readwrite("Data2)", &GUID::Data2)
        .def_readwrite("Data3)", &GUID::Data3)
        .def_property(
            "Data4", [](GUID &self) { return self.Data4; },
            [](GUID &self, unsigned char data4[8]) {
                for (int i = 0; i < 8; i++) {
                    self.Data4[i] = data4[i];
                }
            });

    py::enum_<D3D12_RESOURCE_DIMENSION>(m, "D3D12_RESOURCE_DIMENSION")
        .value("UNKNOWN", D3D12_RESOURCE_DIMENSION_UNKNOWN)
        .value("BUFFER", D3D12_RESOURCE_DIMENSION_BUFFER)
        .value("TEXTURE1D", D3D12_RESOURCE_DIMENSION_TEXTURE1D)
        .value("TEXTURE2D", D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        .value("TEXTURE3D", D3D12_RESOURCE_DIMENSION_TEXTURE3D)
        .export_values();

    py::enum_<D3D12_RESOURCE_FLAGS>(m, "D3D12_RESOURCE_FLAGS", py::arithmetic())
        .value("NONE", D3D12_RESOURCE_FLAG_NONE)
        .value("ALLOW_RENDER_TARGET", D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
        .value("ALLOW_DEPTH_STENCIL", D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
        .value("ALLOW_UNORDERED_ACCESS", D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
        .value("DENY_SHADER_RESOURCE", D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)
        .value("ALLOW_CROSS_ADAPTER", D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER)
        .value("ALLOW_SIMULTANEOUS_ACCESS", D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)
        .value("VIDEO_DECODE_REFERENCE_ONLY", D3D12_RESOURCE_FLAG_VIDEO_DECODE_REFERENCE_ONLY)
        .value("VIDEO_ENCODE_REFERENCE_ONLY", D3D12_RESOURCE_FLAG_VIDEO_ENCODE_REFERENCE_ONLY)
        .value("RAYTRACING_ACCELERATION_STRUCTURE", D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE)
        .export_values();

    py::enum_<D3D12_TEXTURE_LAYOUT>(m, "D3D12_TEXTURE_LAYOUT")
        .value("UNKNOWN", D3D12_TEXTURE_LAYOUT_UNKNOWN)
        .value("ROW_MAJOR", D3D12_TEXTURE_LAYOUT_ROW_MAJOR)
        .value("_64KB_UNDEFINED_SWIZZLE", D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE)
        .value("_64KB_STANDARD_SWIZZLE", D3D12_TEXTURE_LAYOUT_64KB_STANDARD_SWIZZLE)
        .export_values();

    py::class_<D3D12_MIP_REGION>(m, "D3D12_MIP_REGION") //
        .def(py::init())                                //
        .def(py::init([](UINT Width, UINT Height, UINT Depth) {
                 D3D12_MIP_REGION region = {};
                 region.Width            = Width;
                 region.Height           = Height;
                 region.Depth            = Depth;
                 return region;
             }),
             "Width"_a = 0, "Height"_a = 1, "Depth"_a = 1)  //
        .def_readwrite("Width", &D3D12_MIP_REGION::Width)   //
        .def_readwrite("Height", &D3D12_MIP_REGION::Height) //
        .def_readwrite("Depth", &D3D12_MIP_REGION::Depth)   //
        ;

    py::class_<D3D12_RESOURCE_DESC>(m, "D3D12_RESOURCE_DESC") //
        .def(py::init())                                      //
        .def(py::init([](D3D12_RESOURCE_DIMENSION Dimension, UINT64 Alignment, UINT64 Width, UINT Height, UINT16 DepthOrArraySize, UINT16 MipLevels, DXGI_FORMAT Format,
                         DXGI_SAMPLE_DESC SampleDesc, D3D12_TEXTURE_LAYOUT Layout, D3D12_RESOURCE_FLAGS Flags) {
                 D3D12_RESOURCE_DESC desc = {};
                 desc.Dimension           = Dimension;
                 desc.Alignment           = Alignment;
                 desc.Width               = Width;
                 desc.Height              = Height;
                 desc.DepthOrArraySize    = DepthOrArraySize;
                 desc.MipLevels           = MipLevels;
                 desc.Format              = Format;
                 desc.SampleDesc          = SampleDesc;
                 desc.Layout              = Layout;
                 desc.Flags               = Flags;
                 return desc;
             }),
             "Dimension"_a = D3D12_RESOURCE_DIMENSION_BUFFER, "Alignment"_a = 0, "Width"_a = 0, "Height"_a = 1, "DepthOrArraySize"_a = 1, "MipLevels"_a = 1,
             "Format"_a = DXGI_FORMAT_UNKNOWN, "SampleDesc"_a = DXGI_SAMPLE_DESC{1, 0}, "Layout"_a = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, "Flags"_a = D3D12_RESOURCE_FLAG_NONE) //
        .def_readwrite("Dimension", &D3D12_RESOURCE_DESC::Dimension)                                                                                                       //
        .def_readwrite("Alignment", &D3D12_RESOURCE_DESC::Alignment)                                                                                                       //
        .def_readwrite("Width", &D3D12_RESOURCE_DESC::Width)                                                                                                               //
        .def_readwrite("Height", &D3D12_RESOURCE_DESC::Height)                                                                                                             //
        .def_readwrite("DepthOrArraySize", &D3D12_RESOURCE_DESC::DepthOrArraySize)                                                                                         //
        .def_readwrite("MipLevels", &D3D12_RESOURCE_DESC::MipLevels)                                                                                                       //
        .def_readwrite("Format", &D3D12_RESOURCE_DESC::Format)                                                                                                             //
        .def_readwrite("SampleDesc", &D3D12_RESOURCE_DESC::SampleDesc)                                                                                                     //
        .def_readwrite("Layout", &D3D12_RESOURCE_DESC::Layout)                                                                                                             //
        .def_readwrite("Flags", &D3D12_RESOURCE_DESC::Flags)                                                                                                               //
        ;

    py::class_<D3D12_RESOURCE_DESC1>(m, "D3D12_RESOURCE_DESC1") //
        .def(py::init())                                        //
        .def(py::init([](D3D12_RESOURCE_DIMENSION Dimension, UINT64 Alignment, UINT64 Width, UINT Height, UINT16 DepthOrArraySize, UINT16 MipLevels, DXGI_FORMAT Format,
                         DXGI_SAMPLE_DESC SampleDesc, D3D12_TEXTURE_LAYOUT Layout, D3D12_RESOURCE_FLAGS Flags, D3D12_MIP_REGION SamplerFeedbackMipRegion) {
                 D3D12_RESOURCE_DESC1 desc     = {};
                 desc.Dimension                = Dimension;
                 desc.Alignment                = Alignment;
                 desc.Width                    = Width;
                 desc.Height                   = Height;
                 desc.DepthOrArraySize         = DepthOrArraySize;
                 desc.MipLevels                = MipLevels;
                 desc.Format                   = Format;
                 desc.SampleDesc               = SampleDesc;
                 desc.Layout                   = Layout;
                 desc.Flags                    = Flags;
                 desc.SamplerFeedbackMipRegion = SamplerFeedbackMipRegion;
                 return desc;
             }),
             "Dimension"_a = D3D12_RESOURCE_DIMENSION_BUFFER, "Alignment"_a = 0, "Width"_a = 0, "Height"_a = 1, "DepthOrArraySize"_a = 1, "MipLevels"_a = 1,
             "Format"_a = DXGI_FORMAT_UNKNOWN, "SampleDesc"_a = DXGI_SAMPLE_DESC{1, 0}, "Layout"_a = D3D12_TEXTURE_LAYOUT_ROW_MAJOR, "Flags"_a = D3D12_RESOURCE_FLAG_NONE,
             "SamplerFeedbackMipRegion"_a = D3D12_MIP_REGION{})                     //
        .def_readwrite("Dimension", &D3D12_RESOURCE_DESC1::Dimension)               //
        .def_readwrite("Alignment", &D3D12_RESOURCE_DESC1::Alignment)               //
        .def_readwrite("Width", &D3D12_RESOURCE_DESC1::Width)                       //
        .def_readwrite("Height", &D3D12_RESOURCE_DESC1::Height)                     //
        .def_readwrite("DepthOrArraySize", &D3D12_RESOURCE_DESC1::DepthOrArraySize) //
        .def_readwrite("MipLevels", &D3D12_RESOURCE_DESC1::MipLevels)               //
        .def_readwrite("Format", &D3D12_RESOURCE_DESC1::Format)
        .def_readwrite("SampleDesc", &D3D12_RESOURCE_DESC1::SampleDesc)                             //
        .def_readwrite("Layout", &D3D12_RESOURCE_DESC1::Layout)                                     //
        .def_readwrite("Flags", &D3D12_RESOURCE_DESC1::Flags)                                       //
        .def_readwrite("SamplerFeedbackMipRegion", &D3D12_RESOURCE_DESC1::SamplerFeedbackMipRegion) //
        ;
    //

    py::class_<ID3D12ResourceWrapper, std::shared_ptr<ID3D12ResourceWrapper>>(m, "ID3D12Resource")                                  //
        .def(py::init<ID3D12Resource *>())                                                                                          //
        .def("GetGPUVirtualAddress", &ID3D12ResourceWrapper::GetGPUVirtualAddress)                                                  //
        .def("Map", &ID3D12ResourceWrapper::Map, "Subresource"_a = 0, "ReadRange"_a = std::optional<D3D12_RANGE>{std::nullopt})     //
        .def("Unmap", &ID3D12ResourceWrapper::Unmap, "Subresource"_a = 0, "ReadRange"_a = std::optional<D3D12_RANGE>{std::nullopt}) //
        .def("GetDesc", &ID3D12ResourceWrapper::GetDesc)                                                                            //
        .def("SetName", &ID3D12ResourceWrapper::SetName)                                                                            //
        ;

    py::enum_<D3D12_COMMAND_LIST_TYPE>(m, "D3D12_COMMAND_LIST_TYPE")
        .value("DIRECT", D3D12_COMMAND_LIST_TYPE_DIRECT)
        .value("BUNDLE", D3D12_COMMAND_LIST_TYPE_BUNDLE)
        .value("COMPUTE", D3D12_COMMAND_LIST_TYPE_COMPUTE)
        .value("COPY", D3D12_COMMAND_LIST_TYPE_COPY)
        .export_values();

    py::enum_<D3D12_COMMAND_QUEUE_FLAGS>(m, "D3D12_COMMAND_QUEUE_FLAGS")
        .value("NONE", D3D12_COMMAND_QUEUE_FLAG_NONE)
        .value("DISABLE_GPU_TIMEOUT", D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT)
        .export_values();

    py::class_<D3D12_COMMAND_QUEUE_DESC>(m, "D3D12_COMMAND_QUEUE_DESC") //
        .def(py::init())                                                //
        .def(py::init([](D3D12_COMMAND_LIST_TYPE Type, INT Priority, D3D12_COMMAND_QUEUE_FLAGS Flags, UINT NodeMask) {
                 D3D12_COMMAND_QUEUE_DESC desc = {};
                 desc.Type                     = Type;
                 desc.Priority                 = Priority;
                 desc.Flags                    = Flags;
                 desc.NodeMask                 = NodeMask;
                 return desc;
             }),
             "Type"_a = D3D12_COMMAND_LIST_TYPE_DIRECT, "Priority"_a = 0, "Flags"_a = D3D12_COMMAND_QUEUE_FLAG_NONE, "NodeMask"_a = 0) //
        .def_readwrite("Type", &D3D12_COMMAND_QUEUE_DESC::Type)                                                                        //
        .def_readwrite("Priority", &D3D12_COMMAND_QUEUE_DESC::Priority)                                                                //
        .def_readwrite("Flags", &D3D12_COMMAND_QUEUE_DESC::Flags)                                                                      //
        .def_readwrite("NodeMask", &D3D12_COMMAND_QUEUE_DESC::NodeMask)                                                                //
        ;
    py::class_<ID3D12PipelineStateWrapper, std::shared_ptr<ID3D12PipelineStateWrapper>>(m, "ID3D12PipelineState") //
        .def("GetISA", &ID3D12PipelineStateWrapper::GetISA)                                                       //
        ;
    py::class_<ID3D12CommandQueueWrapper, std::shared_ptr<ID3D12CommandQueueWrapper>>(m, "ID3D12CommandQueue")
        .def("ExecuteCommandLists", &ID3D12CommandQueueWrapper::ExecuteCommandLists, "CommandLists"_a) //
        .def("Signal", &ID3D12CommandQueueWrapper::Signal, "Fence"_a, "Value"_a)                       //
        .def("Wait", &ID3D12CommandQueueWrapper::Wait, "Fence"_a, "Value"_a)                           //
        ;

    py::enum_<D3D12_FENCE_FLAGS>(m, "D3D12_FENCE_FLAGS", py::arithmetic())
        .value("NONE", D3D12_FENCE_FLAG_NONE)
        .value("SHARED", D3D12_FENCE_FLAG_SHARED)
        .value("SHARED_CROSS_ADAPTER", D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER)
        .value("NON_MONITORED", D3D12_FENCE_FLAG_NON_MONITORED)
        .export_values();

    py::class_<EventWrapper, std::shared_ptr<EventWrapper>>(m, "Event") //
        .def(py::init())                                                //
        .def("Set", &EventWrapper::Set)                                 //
        .def("Wait", &EventWrapper::Wait)                               //
        .def("Reset", &EventWrapper::Reset)                             //
        ;
    py::class_<ID3D12FenceWrapper, std::shared_ptr<ID3D12FenceWrapper>>(m, "ID3D12Fence")             //
        .def(py::init<ID3D12Fence *>())                                                               //
        .def("GetCompletedValue", &ID3D12FenceWrapper::GetCompletedValue)                             //
        .def("SetEventOnCompletion", &ID3D12FenceWrapper::SetEventOnCompletion, "Value"_a, "Event"_a) //
        .def("Signal", &ID3D12FenceWrapper::Signal, "Value"_a)                                        //
        ;

    py::class_<ID3D12CommandAllocatorWrapper, std::shared_ptr<ID3D12CommandAllocatorWrapper>>(m, "ID3D12CommandAllocator") //
        .def(py::init<ID3D12CommandAllocator *>())                                                                         //
        .def("Reset", &ID3D12CommandAllocatorWrapper::Reset)                                                               //
        ;

    py::class_<D3D12_RESOURCE_TRANSITION_BARRIER>(m, "D3D12_RESOURCE_TRANSITION_BARRIER")                                                 //
        .def(py::init())                                                                                                                  //
        .def(py::init([](std::shared_ptr<ID3D12ResourceWrapper> pResource, UINT Subresource, uint32_t StateBefore, uint32_t StateAfter) { //
                 D3D12_RESOURCE_TRANSITION_BARRIER barrier = {};
                 barrier.pResource                         = pResource->resource;
                 barrier.Subresource                       = Subresource;
                 barrier.StateBefore                       = (D3D12_RESOURCE_STATES)StateBefore;
                 barrier.StateAfter                        = (D3D12_RESOURCE_STATES)StateAfter;
                 return barrier;
             }),
             "Resource"_a, "Subresource"_a = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, "StateBefore"_a, "StateAfter"_a)
        .def_readwrite("Resource", &D3D12_RESOURCE_TRANSITION_BARRIER::pResource)
        .def_readwrite("Subresource", &D3D12_RESOURCE_TRANSITION_BARRIER::Subresource)
        .def_readwrite("StateBefore", &D3D12_RESOURCE_TRANSITION_BARRIER::StateBefore)
        .def_readwrite("StateAfter", &D3D12_RESOURCE_TRANSITION_BARRIER::StateAfter);

    py::class_<D3D12_RESOURCE_ALIASING_BARRIER>(m, "D3D12_RESOURCE_ALIASING_BARRIER")                                                     //
        .def(py::init())                                                                                                                  //
        .def(py::init([](std::shared_ptr<ID3D12ResourceWrapper> pResourceBefore, std::shared_ptr<ID3D12ResourceWrapper> pResourceAfter) { //
                 D3D12_RESOURCE_ALIASING_BARRIER barrier = {};
                 barrier.pResourceBefore                 = pResourceBefore->resource;
                 barrier.pResourceAfter                  = pResourceAfter->resource;
                 return barrier;
             }),
             "ResourceBefore"_a, "ResourceAfter"_a)
        .def_readwrite("ResourceBefore", &D3D12_RESOURCE_ALIASING_BARRIER::pResourceBefore)
        .def_readwrite("ResourceAfter", &D3D12_RESOURCE_ALIASING_BARRIER::pResourceAfter);

    py::class_<D3D12_RESOURCE_UAV_BARRIER>(m, "D3D12_RESOURCE_UAV_BARRIER")  //
        .def(py::init())                                                     //
        .def(py::init([](std::shared_ptr<ID3D12ResourceWrapper> pResource) { //
                 D3D12_RESOURCE_UAV_BARRIER barrier = {};
                 barrier.pResource                  = pResource->resource;
                 return barrier;
             }),
             "Resource"_a)
        .def_readwrite("Resource", &D3D12_RESOURCE_UAV_BARRIER::pResource);

    py::enum_<D3D12_RESOURCE_BARRIER_FLAGS>(m, "D3D12_RESOURCE_BARRIER_FLAGS", py::arithmetic())
        .value("NONE", D3D12_RESOURCE_BARRIER_FLAG_NONE)
        .value("BEGIN_ONLY", D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY)
        .value("END_ONLY", D3D12_RESOURCE_BARRIER_FLAG_END_ONLY)
        .export_values();

    py::enum_<D3D12_RESOURCE_BARRIER_TYPE>(m, "D3D12_RESOURCE_BARRIER_TYPE")
        .value("TRANSITION", D3D12_RESOURCE_BARRIER_TYPE_TRANSITION)
        .value("ALIASING", D3D12_RESOURCE_BARRIER_TYPE_ALIASING)
        .value("UAV", D3D12_RESOURCE_BARRIER_TYPE_UAV)
        .export_values();

    py::class_<D3D12_RESOURCE_BARRIER>(m, "D3D12_RESOURCE_BARRIER")                                         //
        .def(py::init())                                                                                    //
        .def(py::init([](D3D12_RESOURCE_TRANSITION_BARRIER Transition, D3D12_RESOURCE_BARRIER_FLAGS Flag) { //
                 D3D12_RESOURCE_BARRIER barrier = {};
                 barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                 barrier.Flags                  = Flag;
                 barrier.Transition             = Transition;
                 return barrier;
             }),
             "Transition"_a, "Flags"_a = D3D12_RESOURCE_BARRIER_FLAG_NONE)                              //
        .def(py::init([](D3D12_RESOURCE_ALIASING_BARRIER Aliasing, D3D12_RESOURCE_BARRIER_FLAGS Flag) { //
                 D3D12_RESOURCE_BARRIER barrier = {};
                 barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
                 barrier.Flags                  = Flag;
                 barrier.Aliasing               = Aliasing;
                 return barrier;
             }),
             "Aliasing"_a, "Flags"_a = D3D12_RESOURCE_BARRIER_FLAG_NONE)                      //
        .def(py::init([](D3D12_RESOURCE_UAV_BARRIER UAV, D3D12_RESOURCE_BARRIER_FLAGS Flag) { //
                 D3D12_RESOURCE_BARRIER barrier = {};
                 barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                 barrier.Flags                  = Flag;
                 barrier.UAV                    = UAV;
                 return barrier;
             }),
             "UAV"_a, "Flags"_a = D3D12_RESOURCE_BARRIER_FLAG_NONE)       //
        .def_readwrite("Type", &D3D12_RESOURCE_BARRIER::Type)             //
        .def_readwrite("Flags", &D3D12_RESOURCE_BARRIER::Flags)           //
        .def_readwrite("Transition", &D3D12_RESOURCE_BARRIER::Transition) //
        .def_readwrite("Aliasing", &D3D12_RESOURCE_BARRIER::Aliasing)     //
        .def_readwrite("UAV", &D3D12_RESOURCE_BARRIER::UAV)               //
        ;

    py::class_<D3D12_VERTEX_BUFFER_VIEW>(m, "D3D12_VERTEX_BUFFER_VIEW") //
        .def(py::init())                                                //
        .def(py::init([](D3D12_GPU_VIRTUAL_ADDRESS BufferLocation, UINT SizeInBytes, UINT StrideInBytes) {
                 D3D12_VERTEX_BUFFER_VIEW view = {};
                 view.BufferLocation           = BufferLocation;
                 view.SizeInBytes              = SizeInBytes;
                 view.StrideInBytes            = StrideInBytes;
                 return view;
             }),
             "BufferLocation"_a = 0, "SizeInBytes"_a = 0, "StrideInBytes"_a = 0)    //
        .def_readwrite("BufferLocation", &D3D12_VERTEX_BUFFER_VIEW::BufferLocation) //
        .def_readwrite("SizeInBytes", &D3D12_VERTEX_BUFFER_VIEW::SizeInBytes)       //
        .def_readwrite("StrideInBytes", &D3D12_VERTEX_BUFFER_VIEW::StrideInBytes)   //
        ;

    py::class_<D3D12_RECT>(m, "D3D12_RECT") //
        .def(py::init())                    //
        .def(py::init([](LONG left, LONG top, LONG right, LONG bottom) {
                 D3D12_RECT rect = {};
                 rect.left       = left;
                 rect.top        = top;
                 rect.right      = right;
                 rect.bottom     = bottom;
                 return rect;
             }),
             "left"_a = 0, "top"_a = 0, "right"_a = 0, "bottom"_a = 0) //
        .def_readwrite("left", &D3D12_RECT::left)                      //
        .def_readwrite("top", &D3D12_RECT::top)                        //
        .def_readwrite("right", &D3D12_RECT::right)                    //
        .def_readwrite("bottom", &D3D12_RECT::bottom)                  //
        ;
    py::class_<D3D12_VIEWPORT>(m, "D3D12_VIEWPORT") //
        .def(py::init())                            //
        .def(py::init([](FLOAT TopLeftX, FLOAT TopLeftY, FLOAT Width, FLOAT Height, FLOAT MinDepth, FLOAT MaxDepth) {
                 D3D12_VIEWPORT viewport = {};
                 viewport.TopLeftX       = TopLeftX;
                 viewport.TopLeftY       = TopLeftY;
                 viewport.Width          = Width;
                 viewport.Height         = Height;
                 viewport.MinDepth       = MinDepth;
                 viewport.MaxDepth       = MaxDepth;
                 return viewport;
             }),
             "TopLeftX"_a = 0, "TopLeftY"_a = 0, "Width"_a = 0, "Height"_a = 0, "MinDepth"_a = 0, "MaxDepth"_a = 0) //
        .def_readwrite("TopLeftX", &D3D12_VIEWPORT::TopLeftX)                                                       //
        .def_readwrite("TopLeftY", &D3D12_VIEWPORT::TopLeftY)                                                       //
        .def_readwrite("Width", &D3D12_VIEWPORT::Width)                                                             //
        .def_readwrite("Height", &D3D12_VIEWPORT::Height)                                                           //
        .def_readwrite("MinDepth", &D3D12_VIEWPORT::MinDepth)                                                       //
        .def_readwrite("MaxDepth", &D3D12_VIEWPORT::MaxDepth)                                                       //
        ;

    py::class_<D3D12_INDEX_BUFFER_VIEW>(m, "D3D12_INDEX_BUFFER_VIEW") //
        .def(py::init())                                              //
        .def(py::init([](D3D12_GPU_VIRTUAL_ADDRESS BufferLocation, UINT SizeInBytes, DXGI_FORMAT Format) {
                 D3D12_INDEX_BUFFER_VIEW view = {};
                 view.BufferLocation          = BufferLocation;
                 view.SizeInBytes             = SizeInBytes;
                 view.Format                  = Format;
                 return view;
             }),
             "BufferLocation"_a = 0, "SizeInBytes"_a = 0, "Format"_a = DXGI_FORMAT_UNKNOWN) //
        .def_readwrite("BufferLocation", &D3D12_INDEX_BUFFER_VIEW::BufferLocation)          //
        .def_readwrite("SizeInBytes", &D3D12_INDEX_BUFFER_VIEW::SizeInBytes)                //
        .def_readwrite("Format", &D3D12_INDEX_BUFFER_VIEW::Format)                          //
        ;

    // exprt D3D12_TEXTURE_COPY_LOCATION_WRAPPER
    py::class_<D3D12_TEXTURE_COPY_LOCATION_WRAPPER, std::shared_ptr<D3D12_TEXTURE_COPY_LOCATION_WRAPPER>>(m, "D3D12_TEXTURE_COPY_LOCATION") //
        .def(py::init([](std::shared_ptr<ID3D12ResourceWrapper> Resource, uint32_t SubresourceIndex) { return D3D12_TEXTURE_COPY_LOCATION_WRAPPER(Resource, SubresourceIndex); }),
             "Resource"_a, "SubresourceIndex"_a = 0) //
        .def(py::init([](std::shared_ptr<ID3D12ResourceWrapper> Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT PlacedFootprint) {
                 return D3D12_TEXTURE_COPY_LOCATION_WRAPPER(Resource, PlacedFootprint);
             }),
             "Resource"_a, "PlacedFootprint"_a) //

        ;

    // !COMMAND LIST

    py::class_<ID3D12GraphicsCommandListWrapper, std::shared_ptr<ID3D12GraphicsCommandListWrapper>>(m, "ID3D12GraphicsCommandList")                              //
        .def(py::init<ID3D12GraphicsCommandList *>())                                                                                                            //
        .def("Close", &ID3D12GraphicsCommandListWrapper::Close)                                                                                                  //
        .def("Reset", &ID3D12GraphicsCommandListWrapper::Reset, "Allocator"_a, "InitialPSO"_a)                                                                   //
        .def("SetPipelineState", &ID3D12GraphicsCommandListWrapper::SetPipelineState, "PipelineState"_a)                                                         //
        .def("SetComputeRootSignature", &ID3D12GraphicsCommandListWrapper::SetComputeRootSignature, "RootSignature"_a)                                           //
        .def("SetComputeRoot32BitConstant", &ID3D12GraphicsCommandListWrapper::SetComputeRoot32BitConstant, "RootParameterIndex"_a, "SrcData"_a, "DestOffset"_a) //
        .def("SetComputeRoot32BitConstants", &ID3D12GraphicsCommandListWrapper::SetComputeRoot32BitConstants, "RootParameterIndex"_a, "Num32BitValuesToSet"_a, "SrcData"_a,
             "DestOffsetIn32BitValues"_a)                                                                                                                           //
        .def("SetComputeRootConstantBufferView", &ID3D12GraphicsCommandListWrapper::SetComputeRootConstantBufferView, "RootParameterIndex"_a, "BufferLocation"_a)   //
        .def("SetComputeRootShaderResourceView", &ID3D12GraphicsCommandListWrapper::SetComputeRootShaderResourceView, "RootParameterIndex"_a, "BufferLocation"_a)   //
        .def("SetComputeRootUnorderedAccessView", &ID3D12GraphicsCommandListWrapper::SetComputeRootUnorderedAccessView, "RootParameterIndex"_a, "BufferLocation"_a) //
        .def("Dispatch", &ID3D12GraphicsCommandListWrapper::Dispatch, "ThreadGroupCountX"_a, "ThreadGroupCountY"_a, "ThreadGroupCountZ"_a)                          //
        .def("CopyBufferRegion", &ID3D12GraphicsCommandListWrapper::CopyBufferRegion, "DstBuffer"_a, "DstOffset"_a, "SrcBuffer"_a, "SrcOffset"_a, "NumBytes"_a)     //
        .def("CopyTextureRegion", &ID3D12GraphicsCommandListWrapper::CopyTextureRegion, "Dst"_a, "DstX"_a, "DstY"_a, "DstZ"_a, "Src"_a, "SrcBox"_a)                 //
        .def("ResourceBarrier", &ID3D12GraphicsCommandListWrapper::ResourceBarrier, "Barriers"_a)                                                                   //
        .def("IASetVertexBuffers", &ID3D12GraphicsCommandListWrapper::IASetVertexBuffers, "StartSlot"_a, "Views"_a)                                                 //
        .def("IASetIndexBuffer", &ID3D12GraphicsCommandListWrapper::IASetIndexBuffer, "View"_a)                                                                     //
        .def("IASetPrimitiveTopology", &ID3D12GraphicsCommandListWrapper::IASetPrimitiveTopology, "Topology"_a)                                                     //
        .def("DrawIndexedInstanced", &ID3D12GraphicsCommandListWrapper::DrawIndexedInstanced, "IndexCountPerInstance"_a, "InstanceCount"_a, "StartIndexLocation"_a,
             "BaseVertexLocation"_a, "StartInstanceLocation"_a)                                                                                                    //
        .def("SetGraphicsRootSignature", &ID3D12GraphicsCommandListWrapper::SetGraphicsRootSignature, "RootSignature"_a)                                           //
        .def("SetGraphicsRoot32BitConstant", &ID3D12GraphicsCommandListWrapper::SetGraphicsRoot32BitConstant, "RootParameterIndex"_a, "SrcData"_a, "DestOffset"_a) //
        .def("SetGraphicsRoot32BitConstants", &ID3D12GraphicsCommandListWrapper::SetGraphicsRoot32BitConstants, "RootParameterIndex"_a, "Num32BitValuesToSet"_a, "SrcData"_a,
             "DestOffsetIn32BitValues"_a)                                                                                                                             //
        .def("SetGraphicsRootConstantBufferView", &ID3D12GraphicsCommandListWrapper::SetGraphicsRootConstantBufferView, "RootParameterIndex"_a, "BufferLocation"_a)   //
        .def("SetGraphicsRootShaderResourceView", &ID3D12GraphicsCommandListWrapper::SetGraphicsRootShaderResourceView, "RootParameterIndex"_a, "BufferLocation"_a)   //
        .def("SetGraphicsRootUnorderedAccessView", &ID3D12GraphicsCommandListWrapper::SetGraphicsRootUnorderedAccessView, "RootParameterIndex"_a, "BufferLocation"_a) //
        .def("DrawInstanced", &ID3D12GraphicsCommandListWrapper::DrawInstanced, "VertexCountPerInstance"_a, "InstanceCount"_a, "StartVertexLocation"_a,
             "StartInstanceLocation"_a)                                                            //
        .def("RSSetViewports", &ID3D12GraphicsCommandListWrapper::RSSetViewports, "Viewports"_a)   //
        .def("RSSetScissorRects", &ID3D12GraphicsCommandListWrapper::RSSetScissorRects, "Rects"_a) //
        .def("OMSetRenderTargets", &ID3D12GraphicsCommandListWrapper::OMSetRenderTargets, "RenderTargetDescriptors"_a, "RTSingleHandleToDescriptorRange"_a = FALSE,
             "DepthStencilDescriptor"_a = std::optional<D3D12_CPU_DESCRIPTOR_HANDLE>{})                                                                                         //
        .def("ClearRenderTargetView", &ID3D12GraphicsCommandListWrapper::ClearRenderTargetView, "View"_a, "Color"_a, "Rects"_a)                                                 //
        .def("SetDescriptorHeaps", &ID3D12GraphicsCommandListWrapper::SetDescriptorHeaps, "DescriptorHeaps"_a)                                                                  //
        .def("SetGraphicsRootDescriptorTable", &ID3D12GraphicsCommandListWrapper::SetGraphicsRootDescriptorTable, "RootParameterIndex"_a, "BaseDescriptor"_a)                   //
        .def("SetComputeRootDescriptorTable", &ID3D12GraphicsCommandListWrapper::SetComputeRootDescriptorTable, "RootParameterIndex"_a, "BaseDescriptor"_a)                     //
        .def("BuildRaytracingAccelerationStructure", &ID3D12GraphicsCommandListWrapper::BuildRaytracingAccelerationStructure, "Descs"_a, "PostbuildInfoDescs"_a = std::nullopt) //
        .def("EmitRaytracingAccelerationStructurePostbuildInfo", &ID3D12GraphicsCommandListWrapper::EmitRaytracingAccelerationStructurePostbuildInfo, "Descs"_a,
             "SourceAccelerationStructureData"_a) //
        .def("CopyRaytracingAccelerationStructure", &ID3D12GraphicsCommandListWrapper::CopyRaytracingAccelerationStructure, "DestAccelerationStructureData"_a,
             "SourceAccelerationStructureData"_a, "Mode"_a) //
        ;

    py::enum_<D3D12_BLEND_OP>(m, "D3D12_BLEND_OP")
        .value("ADD", D3D12_BLEND_OP_ADD)
        .value("SUBTRACT", D3D12_BLEND_OP_SUBTRACT)
        .value("REV_SUBTRACT", D3D12_BLEND_OP_REV_SUBTRACT)
        .value("MIN", D3D12_BLEND_OP_MIN)
        .value("MAX", D3D12_BLEND_OP_MAX)
        .export_values();

    py::enum_<D3D12_LOGIC_OP>(m, "D3D12_LOGIC_OP")
        .value("CLEAR", D3D12_LOGIC_OP_CLEAR)
        .value("SET", D3D12_LOGIC_OP_SET)
        .value("COPY", D3D12_LOGIC_OP_COPY)
        .value("COPY_INVERTED", D3D12_LOGIC_OP_COPY_INVERTED)
        .value("NOOP", D3D12_LOGIC_OP_NOOP)
        .value("INVERT", D3D12_LOGIC_OP_INVERT)
        .value("AND", D3D12_LOGIC_OP_AND)
        .value("NAND", D3D12_LOGIC_OP_NAND)
        .value("OR", D3D12_LOGIC_OP_OR)
        .value("NOR", D3D12_LOGIC_OP_NOR)
        .value("XOR", D3D12_LOGIC_OP_XOR)
        .value("EQUIV", D3D12_LOGIC_OP_EQUIV)
        .value("AND_REVERSE", D3D12_LOGIC_OP_AND_REVERSE)
        .value("AND_INVERTED", D3D12_LOGIC_OP_AND_INVERTED)
        .value("OR_REVERSE", D3D12_LOGIC_OP_OR_REVERSE)
        .value("OR_INVERTED", D3D12_LOGIC_OP_OR_INVERTED)
        .export_values();

    py::enum_<D3D12_BLEND>(m, "D3D12_BLEND")
        .value("ZERO", D3D12_BLEND_ZERO)
        .value("ONE", D3D12_BLEND_ONE)
        .value("SRC_COLOR", D3D12_BLEND_SRC_COLOR)
        .value("INV_SRC_COLOR", D3D12_BLEND_INV_SRC_COLOR)
        .value("SRC_ALPHA", D3D12_BLEND_SRC_ALPHA)
        .value("INV_SRC_ALPHA", D3D12_BLEND_INV_SRC_ALPHA)
        .value("DEST_ALPHA", D3D12_BLEND_DEST_ALPHA)
        .value("INV_DEST_ALPHA", D3D12_BLEND_INV_DEST_ALPHA)
        .value("DEST_COLOR", D3D12_BLEND_DEST_COLOR)
        .value("INV_DEST_COLOR", D3D12_BLEND_INV_DEST_COLOR)
        .value("SRC_ALPHA_SAT", D3D12_BLEND_SRC_ALPHA_SAT)
        .value("BLEND_FACTOR", D3D12_BLEND_BLEND_FACTOR)
        .value("INV_BLEND_FACTOR", D3D12_BLEND_INV_BLEND_FACTOR)
        .value("SRC1_COLOR", D3D12_BLEND_SRC1_COLOR)
        .value("INV_SRC1_COLOR", D3D12_BLEND_INV_SRC1_COLOR)
        .value("SRC1_ALPHA", D3D12_BLEND_SRC1_ALPHA)
        .value("INV_SRC1_ALPHA", D3D12_BLEND_INV_SRC1_ALPHA)
        .export_values();

    py::enum_<D3D12_COLOR_WRITE_ENABLE>(m, "D3D12_COLOR_WRITE_ENABLE")
        .value("RED", D3D12_COLOR_WRITE_ENABLE_RED)
        .value("GREEN", D3D12_COLOR_WRITE_ENABLE_GREEN)
        .value("BLUE", D3D12_COLOR_WRITE_ENABLE_BLUE)
        .value("ALPHA", D3D12_COLOR_WRITE_ENABLE_ALPHA)
        .value("ALL", D3D12_COLOR_WRITE_ENABLE_ALL)
        .export_values();

    py::class_<D3D12_RENDER_TARGET_BLEND_DESC>(m, "D3D12_RENDER_TARGET_BLEND_DESC") //
        .def(py::init())                                                            //
        .def(py::init([](BOOL BlendEnable, BOOL LogicOpEnable, D3D12_BLEND SrcBlend, D3D12_BLEND DestBlend, D3D12_BLEND_OP BlendOp, D3D12_BLEND SrcBlendAlpha,
                         D3D12_BLEND DestBlendAlpha, D3D12_BLEND_OP BlendOpAlpha, D3D12_LOGIC_OP LogicOp, UINT8 RenderTargetWriteMask) {
                 D3D12_RENDER_TARGET_BLEND_DESC desc = {};
                 desc.BlendEnable                    = BlendEnable;
                 desc.LogicOpEnable                  = LogicOpEnable;
                 desc.SrcBlend                       = SrcBlend;
                 desc.DestBlend                      = DestBlend;
                 desc.BlendOp                        = BlendOp;
                 desc.SrcBlendAlpha                  = SrcBlendAlpha;
                 desc.DestBlendAlpha                 = DestBlendAlpha;
                 desc.BlendOpAlpha                   = BlendOpAlpha;
                 desc.LogicOp                        = LogicOp;
                 desc.RenderTargetWriteMask          = RenderTargetWriteMask;
                 return desc;
             }),
             "BlendEnable"_a = FALSE, "LogicOpEnable"_a = FALSE, "SrcBlend"_a = D3D12_BLEND_ZERO, "DestBlend"_a = D3D12_BLEND_ZERO, "BlendOp"_a = D3D12_BLEND_OP_ADD,
             "SrcBlendAlpha"_a = D3D12_BLEND_ZERO, "DestBlendAlpha"_a = D3D12_BLEND_ZERO, "BlendOpAlpha"_a = D3D12_BLEND_OP_ADD, "LogicOp"_a = D3D12_LOGIC_OP_NOOP,
             "RenderTargetWriteMask"_a = D3D12_COLOR_WRITE_ENABLE_ALL) //
        ;

    py::class_<D3D12_BLEND_DESC>(m, "D3D12_BLEND_DESC") //
        .def(py::init())                                //
        .def(py::init([](BOOL AlphaToCoverageEnable, BOOL IndependentBlendEnable, std::vector<D3D12_RENDER_TARGET_BLEND_DESC> RenderTarget) {
                 D3D12_BLEND_DESC desc       = {};
                 desc.AlphaToCoverageEnable  = AlphaToCoverageEnable;
                 desc.IndependentBlendEnable = IndependentBlendEnable;
                 ASSERT_PANIC(RenderTarget.size() <= D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT &&
                              "RenderTarget size must be less than or equal to D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT");
                 for (int i = 0; i < (int)RenderTarget.size(); i++) {
                     desc.RenderTarget[i] = RenderTarget[i];
                 }
                 return desc;
             }),
             "AlphaToCoverageEnable"_a = FALSE, "IndependentBlendEnable"_a = FALSE,
             "RenderTarget"_a = std::vector<D3D12_RENDER_TARGET_BLEND_DESC>(D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT)) //                                            //
        ;

    py::class_<D3D12_STREAM_OUTPUT_DESC>(m, "D3D12_STREAM_OUTPUT_DESC") //
        .def(py::init())                                                //
        .def(py::init([](std::vector<D3D12_SO_DECLARATION_ENTRY> SODeclaration, std::vector<UINT> BufferStrides, UINT RasterizedStream) {
                 D3D12_STREAM_OUTPUT_DESC desc = {};
                 desc.pSODeclaration           = SODeclaration.data();
                 desc.NumEntries               = (UINT)SODeclaration.size();
                 desc.pBufferStrides           = BufferStrides.data();
                 desc.NumStrides               = (UINT)BufferStrides.size();
                 desc.RasterizedStream         = RasterizedStream;
                 return desc;
             }),
             "SODeclaration"_a = std::vector<D3D12_SO_DECLARATION_ENTRY>{}, "BufferStrides"_a = std::vector<UINT>{},
             "RasterizedStream"_a =
                 0) //                                                                                                                                               //
        ;

    py::enum_<D3D12_DEPTH_WRITE_MASK>(m, "D3D12_DEPTH_WRITE_MASK") //
        .value("ZERO", D3D12_DEPTH_WRITE_MASK_ZERO)                //
        .value("ALL", D3D12_DEPTH_WRITE_MASK_ALL)                  //
        .export_values();
    py::enum_<D3D12_STENCIL_OP>(m, "D3D12_STENCIL_OP") //
        .value("KEEP", D3D12_STENCIL_OP_KEEP)          //
        .value("ZERO", D3D12_STENCIL_OP_ZERO)          //
        .value("REPLACE", D3D12_STENCIL_OP_REPLACE)    //
        .value("INCR_SAT", D3D12_STENCIL_OP_INCR_SAT)  //
        .value("DECR_SAT", D3D12_STENCIL_OP_DECR_SAT)  //
        .value("INVERT", D3D12_STENCIL_OP_INVERT)      //
        .value("INCR", D3D12_STENCIL_OP_INCR)          //
        .value("DECR", D3D12_STENCIL_OP_DECR)          //
        .export_values();

    py::enum_<D3D12_COMPARISON_FUNC>(m, "D3D12_COMPARISON_FUNC")     //
        .value("NEVER", D3D12_COMPARISON_FUNC_NEVER)                 //
        .value("LESS", D3D12_COMPARISON_FUNC_LESS)                   //
        .value("EQUAL", D3D12_COMPARISON_FUNC_EQUAL)                 //
        .value("LESS_EQUAL", D3D12_COMPARISON_FUNC_LESS_EQUAL)       //
        .value("GREATER", D3D12_COMPARISON_FUNC_GREATER)             //
        .value("NOT_EQUAL", D3D12_COMPARISON_FUNC_NOT_EQUAL)         //
        .value("GREATER_EQUAL", D3D12_COMPARISON_FUNC_GREATER_EQUAL) //
        .value("ALWAYS", D3D12_COMPARISON_FUNC_ALWAYS)               //
        .export_values();

    py::class_<D3D12_DEPTH_STENCILOP_DESC>(m, "D3D12_DEPTH_STENCILOP_DESC") //
        .def(py::init())                                                    //
        .def(py::init([](D3D12_STENCIL_OP StencilFailOp, D3D12_STENCIL_OP StencilDepthFailOp, D3D12_STENCIL_OP StencilPassOp, D3D12_COMPARISON_FUNC StencilFunc) {
                 D3D12_DEPTH_STENCILOP_DESC desc = {};
                 desc.StencilFailOp              = StencilFailOp;
                 desc.StencilDepthFailOp         = StencilDepthFailOp;
                 desc.StencilPassOp              = StencilPassOp;
                 desc.StencilFunc                = StencilFunc;
                 return desc;
             }),
             "StencilFailOp"_a = D3D12_STENCIL_OP_KEEP, "StencilDepthFailOp"_a = D3D12_STENCIL_OP_KEEP, "StencilPassOp"_a = D3D12_STENCIL_OP_KEEP,
             "StencilFunc"_a = D3D12_COMPARISON_FUNC_ALWAYS) //
                                                             //
        ;

    py::class_<D3D12_DEPTH_STENCIL_DESC>(m, "D3D12_DEPTH_STENCIL_DESC") //
        .def(py::init())                                                //
        .def(py::init([](BOOL DepthEnable, D3D12_DEPTH_WRITE_MASK DepthWriteMask, D3D12_COMPARISON_FUNC DepthFunc, BOOL StencilEnable, UINT8 StencilReadMask,
                         UINT8 StencilWriteMask, D3D12_DEPTH_STENCILOP_DESC FrontFace, D3D12_DEPTH_STENCILOP_DESC BackFace) {
                 D3D12_DEPTH_STENCIL_DESC desc = {};
                 desc.DepthEnable              = DepthEnable;
                 desc.DepthWriteMask           = DepthWriteMask;
                 desc.DepthFunc                = DepthFunc;
                 desc.StencilEnable            = StencilEnable;
                 desc.StencilReadMask          = StencilReadMask;
                 desc.StencilWriteMask         = StencilWriteMask;
                 desc.FrontFace                = FrontFace;
                 desc.BackFace                 = BackFace;
                 return desc;
             }),
             "DepthEnable"_a = FALSE, "DepthWriteMask"_a = D3D12_DEPTH_WRITE_MASK_ALL, "DepthFunc"_a = D3D12_COMPARISON_FUNC_LESS, "StencilEnable"_a = FALSE,
             "StencilReadMask"_a = D3D12_DEFAULT_STENCIL_READ_MASK, "StencilWriteMask"_a = D3D12_DEFAULT_STENCIL_WRITE_MASK, "FrontFace"_a = D3D12_DEPTH_STENCILOP_DESC{},
             "BackFace"_a = D3D12_DEPTH_STENCILOP_DESC{}) //
        ;

    py::enum_<D3D12_CONSERVATIVE_RASTERIZATION_MODE>(m, "D3D12_CONSERVATIVE_RASTERIZATION_MODE") //
        .value("OFF", D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF)                                 //
        .value("ON", D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON)                                   //
        .export_values();
    py::enum_<D3D12_CULL_MODE>(m, "D3D12_CULL_MODE") //
        .value("NONE", D3D12_CULL_MODE_NONE)         //
        .value("FRONT", D3D12_CULL_MODE_FRONT)       //
        .value("BACK", D3D12_CULL_MODE_BACK)         //
        .export_values();
    py::enum_<D3D12_FILL_MODE>(m, "D3D12_FILL_MODE")   //
        .value("WIREFRAME", D3D12_FILL_MODE_WIREFRAME) //
        .value("SOLID", D3D12_FILL_MODE_SOLID)         //
        .export_values();
    py::class_<D3D12_RASTERIZER_DESC>(m, "D3D12_RASTERIZER_DESC") //
        .def(py::init())                                          //
        .def(py::init([](D3D12_FILL_MODE FillMode, D3D12_CULL_MODE CullMode, BOOL FrontCounterClockwise, INT DepthBias, FLOAT DepthBiasClamp, FLOAT SlopeScaledDepthBias,
                         BOOL DepthClipEnable, BOOL MultisampleEnable, BOOL AntialiasedLineEnable, UINT ForcedSampleCount,
                         D3D12_CONSERVATIVE_RASTERIZATION_MODE ConservativeRaster) {
                 D3D12_RASTERIZER_DESC desc = {};
                 desc.FillMode              = FillMode;
                 desc.CullMode              = CullMode;
                 desc.FrontCounterClockwise = FrontCounterClockwise;
                 desc.DepthBias             = DepthBias;
                 desc.DepthBiasClamp        = DepthBiasClamp;
                 desc.SlopeScaledDepthBias  = SlopeScaledDepthBias;
                 desc.DepthClipEnable       = DepthClipEnable;
                 desc.MultisampleEnable     = MultisampleEnable;
                 desc.AntialiasedLineEnable = AntialiasedLineEnable;
                 desc.ForcedSampleCount     = ForcedSampleCount;
                 desc.ConservativeRaster    = ConservativeRaster;
                 return desc;
             }),
             "FillMode"_a = D3D12_FILL_MODE_SOLID, "CullMode"_a = D3D12_CULL_MODE_BACK, "FrontCounterClockwise"_a = FALSE, "DepthBias"_a = D3D12_DEFAULT_DEPTH_BIAS,
             "DepthBiasClamp"_a = D3D12_DEFAULT_DEPTH_BIAS_CLAMP, "SlopeScaledDepthBias"_a = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS, "DepthClipEnable"_a = TRUE,
             "MultisampleEnable"_a = FALSE, "AntialiasedLineEnable"_a = FALSE, "ForcedSampleCount"_a = 0, "ConservativeRaster"_a = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF) //
        ;

    py::enum_<D3D12_INPUT_CLASSIFICATION>(m, "D3D12_INPUT_CLASSIFICATION")        //
        .value("PER_VERTEX_DATA", D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA)     //
        .value("PER_INSTANCE_DATA", D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA) //
        .export_values();
    py::class_<D3D12_INPUT_ELEMENT_DESC_Wrapper, std::shared_ptr<D3D12_INPUT_ELEMENT_DESC_Wrapper>>(m, "D3D12_INPUT_ELEMENT_DESC") //
        .def(py::init())                                                                                                           //
        .def(py::init([](std::string SemanticName, UINT SemanticIndex, DXGI_FORMAT Format, UINT InputSlot, UINT AlignedByteOffset, D3D12_INPUT_CLASSIFICATION InputSlotClass,
                         UINT InstanceDataStepRate) {
                 return D3D12_INPUT_ELEMENT_DESC_Wrapper(SemanticName, SemanticIndex, Format, InputSlot, AlignedByteOffset, InputSlotClass, InstanceDataStepRate);
             }),
             "SemanticName"_a = "", "SemanticIndex"_a = 0, "Format"_a = DXGI_FORMAT_UNKNOWN, "InputSlot"_a = 0, "AlignedByteOffset"_a = D3D12_APPEND_ALIGNED_ELEMENT,
             "InputSlotClass"_a = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, "InstanceDataStepRate"_a = 0) //
        ;
    py::enum_<D3D12_PRIMITIVE_TOPOLOGY>(m, "D3D12_PRIMITIVE_TOPOLOGY") //
        .value("UNDEFINED", D3D_PRIMITIVE_TOPOLOGY_UNDEFINED)          //
        .value("POINTLIST", D3D_PRIMITIVE_TOPOLOGY_POINTLIST)          //
        .value("LINELIST", D3D_PRIMITIVE_TOPOLOGY_LINELIST)            //
        .value("LINESTRIP", D3D_PRIMITIVE_TOPOLOGY_LINESTRIP)          //
        .value("TRIANGLELIST", D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST)    //
        .value("TRIANGLESTRIP", D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP)  //
        .export_values();
    py::enum_<D3D12_PRIMITIVE_TOPOLOGY_TYPE>(m, "D3D12_PRIMITIVE_TOPOLOGY_TYPE") //
        .value("UNDEFINED", D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED)             //
        .value("POINT", D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT)                     //
        .value("LINE", D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE)                       //
        .value("TRIANGLE", D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE)               //
        .value("PATCH", D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH)                     //
        .export_values();
    py::enum_<D3D12_INDEX_BUFFER_STRIP_CUT_VALUE>(m, "D3D12_INDEX_BUFFER_STRIP_CUT_VALUE") //
        .value("DISABLED", D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED)                    //
        .value("FFFF", D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF)                          //
        .value("FFFFFFFF", D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF)                  //
        .export_values();

    py::class_<D3D12_GRAPHICS_PIPELINE_STATE_DESC_Wrapper, std::shared_ptr<D3D12_GRAPHICS_PIPELINE_STATE_DESC_Wrapper>>(m, "D3D12_GRAPHICS_PIPELINE_STATE_DESC")
        .def(py::init([](std::shared_ptr<ID3D12RootSignatureWrapper>                    pRootSignature,        //
                         std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper>                 VS,                    //
                         std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper>                 PS,                    //
                         std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper>                 DS,                    //
                         std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper>                 HS,                    //
                         std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper>                 GS,                    //
                         std::optional<D3D12_STREAM_OUTPUT_DESC>                        StreamOutput,          //
                         D3D12_BLEND_DESC                                               BlendState,            //
                         UINT                                                           SampleMask,            //
                         D3D12_RASTERIZER_DESC                                          RasterizerState,       //
                         D3D12_DEPTH_STENCIL_DESC                                       DepthStencilState,     //
                         std::vector<std::shared_ptr<D3D12_INPUT_ELEMENT_DESC_Wrapper>> InputLayouts,          //
                         D3D12_INDEX_BUFFER_STRIP_CUT_VALUE                             IBStripCutValue,       //
                         D3D12_PRIMITIVE_TOPOLOGY_TYPE                                  PrimitiveTopologyType, //
                         std::vector<DXGI_FORMAT>                                       RTVFormats,            //
                         DXGI_FORMAT                                                    DSVFormat,             //
                         DXGI_SAMPLE_DESC                                               SampleDesc,            //
                         UINT                                                           NodeMask,              //
                         std::optional<D3D12_CACHED_PIPELINE_STATE>                     CachedPSO,             //
                         D3D12_PIPELINE_STATE_FLAGS                                     Flags) {                                                   //
                 UINT NumRenderTargets = (UINT)RTVFormats.size();
                 ASSERT_PANIC(NumRenderTargets <= D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT &&
                              "NumRenderTargets must be less than or equal to D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT");
                 return D3D12_GRAPHICS_PIPELINE_STATE_DESC_Wrapper(pRootSignature, VS, PS, DS, HS, GS, StreamOutput ? *StreamOutput : D3D12_STREAM_OUTPUT_DESC{}, BlendState,
                                                                   SampleMask, RasterizerState, DepthStencilState, InputLayouts, IBStripCutValue, PrimitiveTopologyType,
                                                                   NumRenderTargets, RTVFormats, DSVFormat, SampleDesc, NodeMask, CachedPSO, Flags);

             }), //
             "RootSignature"_a = nullptr, "VS"_a = nullptr, "PS"_a = nullptr, "DS"_a = nullptr, "HS"_a = nullptr, "GS"_a = nullptr,
             "StreamOutput"_a = std::optional<D3D12_STREAM_OUTPUT_DESC>{}, "BlendState"_a = D3D12_BLEND_DESC{}, "SampleMask"_a = 1, "RasterizerState"_a = D3D12_RASTERIZER_DESC{},
             "DepthStencilState"_a = D3D12_DEPTH_STENCIL_DESC{}, "InputLayouts"_a = std::vector<std::shared_ptr<D3D12_INPUT_ELEMENT_DESC_Wrapper>>{},
             "IBStripCutValue"_a = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE{}, "PrimitiveTopologyType"_a = D3D12_PRIMITIVE_TOPOLOGY_TYPE{}, "RTVFormats"_a = std::array<DXGI_FORMAT, 8>{},
             "DSVFormat"_a = DXGI_FORMAT_UNKNOWN, "SampleDesc"_a = DXGI_SAMPLE_DESC{}, "NodeMask"_a = 0, "CachedPSO"_a = std::optional<D3D12_CACHED_PIPELINE_STATE>{},
             "Flags"_a = D3D12_PIPELINE_STATE_FLAG_NONE) //
        ;

    py::enum_<D3D12_DESCRIPTOR_HEAP_TYPE>(m, "D3D12_DESCRIPTOR_HEAP_TYPE") //
        .value("CBV_SRV_UAV", D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)      //
        .value("SAMPLER", D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER)              //
        .value("RTV", D3D12_DESCRIPTOR_HEAP_TYPE_RTV)                      //
        .value("DSV", D3D12_DESCRIPTOR_HEAP_TYPE_DSV)                      //
        .export_values();

    py::enum_<D3D12_DESCRIPTOR_HEAP_FLAGS>(m, "D3D12_DESCRIPTOR_HEAP_FLAGS") //
        .value("NONE", D3D12_DESCRIPTOR_HEAP_FLAG_NONE)                      //
        .value("SHADER_VISIBLE", D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)  //
        .export_values();

    py::class_<D3D12_DESCRIPTOR_HEAP_DESC>(m, "D3D12_DESCRIPTOR_HEAP_DESC") //
        .def(py::init())                                                    //
        .def(py::init([](D3D12_DESCRIPTOR_HEAP_TYPE Type, UINT NumDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS Flags, UINT NodeMask) {
                 D3D12_DESCRIPTOR_HEAP_DESC desc = {};
                 desc.Type                       = Type;
                 desc.NumDescriptors             = NumDescriptors;
                 desc.Flags                      = Flags;
                 desc.NodeMask                   = NodeMask;
                 return desc;
             }),
             "Type"_a = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, "NumDescriptors"_a = 1, "Flags"_a = D3D12_DESCRIPTOR_HEAP_FLAG_NONE, "NodeMask"_a = 0) //
        ;

    // export D3D12_TEX1D_RTV
    py::class_<D3D12_TEX1D_RTV>(m, "D3D12_TEX1D_RTV") //
        .def(py::init())                              //
        .def(py::init([](UINT MipSlice) {
                 D3D12_TEX1D_RTV desc = {};
                 desc.MipSlice        = MipSlice;
                 return desc;
             }),
             "MipSlice"_a = 0)                                 //
        .def_readwrite("MipSlice", &D3D12_TEX1D_RTV::MipSlice) //
        ;

    // export D3D12_TEX1D_ARRAY_RTV
    py::class_<D3D12_TEX1D_ARRAY_RTV>(m, "D3D12_TEX1D_ARRAY_RTV") //
        .def(py::init())                                          //
        .def(py::init([](UINT MipSlice, UINT FirstArraySlice, UINT ArraySize) {
                 D3D12_TEX1D_ARRAY_RTV desc = {};
                 desc.MipSlice              = MipSlice;
                 desc.FirstArraySlice       = FirstArraySlice;
                 desc.ArraySize             = ArraySize;
                 return desc;
             }),
             "MipSlice"_a = 0, "FirstArraySlice"_a = 0, "ArraySize"_a = 1)         //
        .def_readwrite("MipSlice", &D3D12_TEX1D_ARRAY_RTV::MipSlice)               //
        .def_readwrite("FirstArraySlice", &D3D12_TEX1D_ARRAY_RTV::FirstArraySlice) //
        .def_readwrite("ArraySize", &D3D12_TEX1D_ARRAY_RTV::ArraySize)             //
        ;

    // export D3D12_TEX2D_RTV
    py::class_<D3D12_TEX2D_RTV>(m, "D3D12_TEX2D_RTV") //
        .def(py::init())                              //
        .def(py::init([](UINT MipSlice, UINT PlaneSlice) {
                 D3D12_TEX2D_RTV desc = {};
                 desc.MipSlice        = MipSlice;
                 desc.PlaneSlice      = PlaneSlice;
                 return desc;
             }),
             "MipSlice"_a = 0, "PlaneSlice"_a = 0)                 //
        .def_readwrite("MipSlice", &D3D12_TEX2D_RTV::MipSlice)     //
        .def_readwrite("PlaneSlice", &D3D12_TEX2D_RTV::PlaneSlice) //
        ;

    // export D3D12_TEX2D_ARRAY_RTV
    py::class_<D3D12_TEX2D_ARRAY_RTV>(m, "D3D12_TEX2D_ARRAY_RTV") //
        .def(py::init())                                          //
        .def(py::init([](UINT MipSlice, UINT FirstArraySlice, UINT ArraySize, UINT PlaneSlice) {
                 D3D12_TEX2D_ARRAY_RTV desc = {};
                 desc.MipSlice              = MipSlice;
                 desc.FirstArraySlice       = FirstArraySlice;
                 desc.ArraySize             = ArraySize;
                 desc.PlaneSlice            = PlaneSlice;
                 return desc;
             }),
             "MipSlice"_a = 0, "FirstArraySlice"_a = 0, "ArraySize"_a = 1, "PlaneSlice"_a = 0) //
        .def_readwrite("MipSlice", &D3D12_TEX2D_ARRAY_RTV::MipSlice)                           //
        .def_readwrite("FirstArraySlice", &D3D12_TEX2D_ARRAY_RTV::FirstArraySlice)             //
        .def_readwrite("ArraySize", &D3D12_TEX2D_ARRAY_RTV::ArraySize)                         //
        .def_readwrite("PlaneSlice", &D3D12_TEX2D_ARRAY_RTV::PlaneSlice)                       //
        ;

    // export D3D12_TEX2DMS_RTV
    py::class_<D3D12_TEX2DMS_RTV>(m, "D3D12_TEX2DMS_RTV") //
        .def(py::init())                                  //
        .def(py::init([](UINT UnusedField_NothingToDefine) {
                 D3D12_TEX2DMS_RTV desc           = {};
                 desc.UnusedField_NothingToDefine = UnusedField_NothingToDefine;
                 return desc;
             }),
             "UnusedField_NothingToDefine"_a = 0)                                                      //
        .def_readwrite("UnusedField_NothingToDefine", &D3D12_TEX2DMS_RTV::UnusedField_NothingToDefine) //
        ;

    // export D3D12_TEX2DMS_ARRAY_RTV
    py::class_<D3D12_TEX2DMS_ARRAY_RTV>(m, "D3D12_TEX2DMS_ARRAY_RTV") //
        .def(py::init())                                              //
        .def(py::init([](UINT FirstArraySlice, UINT ArraySize) {
                 D3D12_TEX2DMS_ARRAY_RTV desc = {};
                 desc.FirstArraySlice         = FirstArraySlice;
                 desc.ArraySize               = ArraySize;
                 return desc;
             }),
             "FirstArraySlice"_a = 0, "ArraySize"_a = 1)                             //
        .def_readwrite("FirstArraySlice", &D3D12_TEX2DMS_ARRAY_RTV::FirstArraySlice) //
        .def_readwrite("ArraySize", &D3D12_TEX2DMS_ARRAY_RTV::ArraySize)             //
        ;

    // export D3D12_TEX3D_RTV
    py::class_<D3D12_TEX3D_RTV>(m, "D3D12_TEX3D_RTV") //
        .def(py::init())                              //
        .def(py::init([](UINT MipSlice, UINT FirstWSlice, UINT WSize) {
                 D3D12_TEX3D_RTV desc = {};
                 desc.MipSlice        = MipSlice;
                 desc.FirstWSlice     = FirstWSlice;
                 desc.WSize           = WSize;
                 return desc;
             }),
             "MipSlice"_a = 0, "FirstWSlice"_a = 0, "WSize"_a = 1)   //
        .def_readwrite("MipSlice", &D3D12_TEX3D_RTV::MipSlice)       //
        .def_readwrite("FirstWSlice", &D3D12_TEX3D_RTV::FirstWSlice) //
        .def_readwrite("WSize", &D3D12_TEX3D_RTV::WSize)             //
        ;

    // export D3D12_BUFFER_RTV
    py::class_<D3D12_BUFFER_RTV>(m, "D3D12_BUFFER_RTV") //
        .def(py::init())                                //
        .def(py::init([](UINT64 FirstElement, UINT NumElements) {
                 D3D12_BUFFER_RTV desc = {};
                 desc.FirstElement     = FirstElement;
                 desc.NumElements      = NumElements;
                 return desc;
             }),
             "FirstElement"_a = 0, "NumElements"_a = 1)                 //
        .def_readwrite("FirstElement", &D3D12_BUFFER_RTV::FirstElement) //
        .def_readwrite("NumElements", &D3D12_BUFFER_RTV::NumElements)   //
        ;

    py::enum_<D3D12_RTV_DIMENSION>(m, "D3D12_RTV_DIMENSION")             //
        .value("UNKNOWN", D3D12_RTV_DIMENSION_UNKNOWN)                   //
        .value("BUFFER", D3D12_RTV_DIMENSION_BUFFER)                     //
        .value("TEXTURE1D", D3D12_RTV_DIMENSION_TEXTURE1D)               //
        .value("TEXTURE1DARRAY", D3D12_RTV_DIMENSION_TEXTURE1DARRAY)     //
        .value("TEXTURE2D", D3D12_RTV_DIMENSION_TEXTURE2D)               //
        .value("TEXTURE2DARRAY", D3D12_RTV_DIMENSION_TEXTURE2DARRAY)     //
        .value("TEXTURE2DMS", D3D12_RTV_DIMENSION_TEXTURE2DMS)           //
        .value("TEXTURE2DMSARRAY", D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY) //
        .value("TEXTURE3D", D3D12_RTV_DIMENSION_TEXTURE3D)               //
        .export_values();

    py::class_<D3D12_RENDER_TARGET_VIEW_DESC>(m, "D3D12_RENDER_TARGET_VIEW_DESC") //

        .def(py::init([](DXGI_FORMAT Format, D3D12_TEX1D_RTV Texture1D) {
                 D3D12_RENDER_TARGET_VIEW_DESC desc = {};
                 desc.Format                        = Format;
                 desc.ViewDimension                 = D3D12_RTV_DIMENSION_TEXTURE1D;
                 desc.Texture1D                     = Texture1D;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Texture1D"_a = D3D12_TEX1D_RTV{}) //
        .def(py::init([](DXGI_FORMAT Format, D3D12_TEX1D_ARRAY_RTV Texture1DArray) {
                 D3D12_RENDER_TARGET_VIEW_DESC desc = {};
                 desc.Format                        = Format;
                 desc.ViewDimension                 = D3D12_RTV_DIMENSION_TEXTURE1DARRAY;
                 desc.Texture1DArray                = Texture1DArray;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Texture1DArray"_a = D3D12_TEX1D_ARRAY_RTV{}) //
        .def(py::init([](DXGI_FORMAT Format, D3D12_TEX2D_RTV Texture2D) {
                 D3D12_RENDER_TARGET_VIEW_DESC desc = {};
                 desc.Format                        = Format;
                 desc.ViewDimension                 = D3D12_RTV_DIMENSION_TEXTURE2D;
                 desc.Texture2D                     = Texture2D;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Texture2D"_a = D3D12_TEX2D_RTV{}) //
        .def(py::init([](DXGI_FORMAT Format, D3D12_TEX2D_ARRAY_RTV Texture2DArray) {
                 D3D12_RENDER_TARGET_VIEW_DESC desc = {};
                 desc.Format                        = Format;
                 desc.ViewDimension                 = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
                 desc.Texture2DArray                = Texture2DArray;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Texture2DArray"_a = D3D12_TEX2D_ARRAY_RTV{}) //
        .def(py::init([](DXGI_FORMAT Format, D3D12_TEX2DMS_RTV Texture2DMS) {
                 D3D12_RENDER_TARGET_VIEW_DESC desc = {};
                 desc.Format                        = Format;
                 desc.ViewDimension                 = D3D12_RTV_DIMENSION_TEXTURE2DMS;
                 desc.Texture2DMS                   = Texture2DMS;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Texture2DMS"_a = D3D12_TEX2DMS_RTV{}) //
        .def(py::init([](DXGI_FORMAT Format, D3D12_TEX2DMS_ARRAY_RTV Texture2DMSArray) {
                 D3D12_RENDER_TARGET_VIEW_DESC desc = {};
                 desc.Format                        = Format;
                 desc.ViewDimension                 = D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
                 desc.Texture2DMSArray              = Texture2DMSArray;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Texture2DMSArray"_a = D3D12_TEX2DMS_ARRAY_RTV{}) //
        .def(py::init([](DXGI_FORMAT Format, D3D12_TEX3D_RTV Texture3D) {
                 D3D12_RENDER_TARGET_VIEW_DESC desc = {};
                 desc.Format                        = Format;
                 desc.ViewDimension                 = D3D12_RTV_DIMENSION_TEXTURE3D;
                 desc.Texture3D                     = Texture3D;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Texture3D"_a = D3D12_TEX3D_RTV{}) //
        .def(py::init([](DXGI_FORMAT Format, D3D12_BUFFER_RTV Buffer) {
                 D3D12_RENDER_TARGET_VIEW_DESC desc = {};
                 desc.Format                        = Format;
                 desc.ViewDimension                 = D3D12_RTV_DIMENSION_BUFFER;
                 desc.Buffer                        = Buffer;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Buffer"_a = D3D12_BUFFER_RTV{}) //
        ;

    py::class_<D3D12_GPU_DESCRIPTOR_HANDLE>(m, "D3D12_GPU_DESCRIPTOR_HANDLE")                        //
        .def(py::init())                                                                             //
        .def(py::init([](uint64_t handle) { return D3D12_GPU_DESCRIPTOR_HANDLE{handle}; }), "ptr"_a) //
        .def_readwrite("ptr", &D3D12_GPU_DESCRIPTOR_HANDLE::ptr)                                     //
        ;
    py::class_<D3D12_CPU_DESCRIPTOR_HANDLE>(m, "D3D12_CPU_DESCRIPTOR_HANDLE")                        //
        .def(py::init())                                                                             //
        .def(py::init([](uint64_t handle) { return D3D12_CPU_DESCRIPTOR_HANDLE{handle}; }), "ptr"_a) //
        .def_readwrite("ptr", &D3D12_CPU_DESCRIPTOR_HANDLE::ptr)                                     //
        ;

    py::class_<D3D12_SUBRESOURCE_FOOTPRINT>(m, "D3D12_SUBRESOURCE_FOOTPRINT") //
        .def(py::init())                                                      //
        .def_readwrite("Format", &D3D12_SUBRESOURCE_FOOTPRINT::Format)        //
        .def_readwrite("Width", &D3D12_SUBRESOURCE_FOOTPRINT::Width)          //
        .def_readwrite("Height", &D3D12_SUBRESOURCE_FOOTPRINT::Height)        //
        .def_readwrite("Depth", &D3D12_SUBRESOURCE_FOOTPRINT::Depth)          //
        .def_readwrite("RowPitch", &D3D12_SUBRESOURCE_FOOTPRINT::RowPitch)    //
        ;
    py::class_<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>(m, "D3D12_PLACED_SUBRESOURCE_FOOTPRINT") //
        .def(py::init())                                                                    //
        .def_readwrite("Offset", &D3D12_PLACED_SUBRESOURCE_FOOTPRINT::Offset)               //
        .def_readwrite("Footprint", &D3D12_PLACED_SUBRESOURCE_FOOTPRINT::Footprint)         //
        ;
    py::class_<ID3D12DescriptorHeapWrapper, std::shared_ptr<ID3D12DescriptorHeapWrapper>>(m, "ID3D12DescriptorHeap") //
        .def(py::init<ID3D12DescriptorHeap *>())                                                                     //
        .def("GetCPUDescriptorHandleForHeapStart", &ID3D12DescriptorHeapWrapper::GetCPUDescriptorHandleForHeapStart) //
        .def("GetGPUDescriptorHandleForHeapStart", &ID3D12DescriptorHeapWrapper::GetGPUDescriptorHandleForHeapStart) //                                                    //
        ;
    py::class_<CopyableFootprints, std::shared_ptr<CopyableFootprints>>(m, "CopyableFootprints")
        .def(py::init())                                                      //
        .def_readwrite("Layouts", &CopyableFootprints::layouts)               //
        .def_readwrite("NumRows", &CopyableFootprints::NumRows)               //
        .def_readwrite("RowSizeInBytes", &CopyableFootprints::RowSizeInBytes) //
        .def_readwrite("TotalBytes", &CopyableFootprints::TotalBytes)         //
        ;

    py::enum_<D3D12_BUFFER_SRV_FLAGS>(m, "D3D12_BUFFER_SRV_FLAGS", py::arithmetic()) //
        .value("NONE", D3D12_BUFFER_SRV_FLAG_NONE)                                   //
        .value("RAW", D3D12_BUFFER_SRV_FLAG_RAW)                                     //
        .export_values();
    py::class_<D3D12_BUFFER_SRV>(m, "D3D12_BUFFER_SRV") //
        .def(py::init([](UINT64 FirstElement, UINT NumElements, UINT StructureByteStride, D3D12_BUFFER_SRV_FLAGS Flags) {
                 D3D12_BUFFER_SRV desc    = {};
                 desc.FirstElement        = FirstElement;
                 desc.NumElements         = NumElements;
                 desc.StructureByteStride = StructureByteStride;
                 desc.Flags               = Flags;
                 return desc;
             }),
             "FirstElement"_a = 0, "NumElements"_a = 1, "StructureByteStride"_a = 0, "Flags"_a = D3D12_BUFFER_SRV_FLAG_NONE) //
        ;

    py::class_<D3D12_TEX1D_SRV>(m, "D3D12_TEX1D_SRV") //
        .def(py::init([](UINT MostDetailedMip, UINT MipLevels, FLOAT ResourceMinLODClamp) {
                 D3D12_TEX1D_SRV desc     = {};
                 desc.MostDetailedMip     = MostDetailedMip;
                 desc.MipLevels           = MipLevels;
                 desc.ResourceMinLODClamp = ResourceMinLODClamp;
                 return desc;
             }),
             "MostDetailedMip"_a = 0, "MipLevels"_a = -1, "ResourceMinLODClamp"_a = 0.0f) //
        ;
    py::class_<D3D12_TEX1D_ARRAY_SRV>(m, "D3D12_TEX1D_ARRAY_SRV") //
        .def(py::init([](UINT MostDetailedMip, UINT MipLevels, UINT FirstArraySlice, UINT ArraySize, FLOAT ResourceMinLODClamp) {
                 D3D12_TEX1D_ARRAY_SRV desc = {};
                 desc.MostDetailedMip       = MostDetailedMip;
                 desc.MipLevels             = MipLevels;
                 desc.FirstArraySlice       = FirstArraySlice;
                 desc.ArraySize             = ArraySize;
                 desc.ResourceMinLODClamp   = ResourceMinLODClamp;
                 return desc;
             }),
             "MostDetailedMip"_a = 0, "MipLevels"_a = -1, "FirstArraySlice"_a = 0, "ArraySize"_a = 1, "ResourceMinLODClamp"_a = 0.0f) //
        ;
    py::class_<D3D12_TEX2D_SRV>(m, "D3D12_TEX2D_SRV") //
        .def(py::init([](UINT MostDetailedMip, UINT MipLevels, UINT PlaneSlice, FLOAT ResourceMinLODClamp) {
                 D3D12_TEX2D_SRV desc     = {};
                 desc.MostDetailedMip     = MostDetailedMip;
                 desc.MipLevels           = MipLevels;
                 desc.PlaneSlice          = PlaneSlice;
                 desc.ResourceMinLODClamp = ResourceMinLODClamp;
                 return desc;
             }),
             "MostDetailedMip"_a = 0, "MipLevels"_a = -1, "PlaneSlice"_a = 0, "ResourceMinLODClamp"_a = 0.0f) //
        ;
    py::class_<D3D12_TEX2D_ARRAY_SRV>(m, "D3D12_TEX2D_ARRAY_SRV") //
        .def(py::init([](UINT MostDetailedMip, UINT MipLevels, UINT FirstArraySlice, UINT ArraySize, UINT PlaneSlice, FLOAT ResourceMinLODClamp) {
                 D3D12_TEX2D_ARRAY_SRV desc = {};
                 desc.MostDetailedMip       = MostDetailedMip;
                 desc.MipLevels             = MipLevels;
                 desc.FirstArraySlice       = FirstArraySlice;
                 desc.ArraySize             = ArraySize;
                 desc.PlaneSlice            = PlaneSlice;
                 desc.ResourceMinLODClamp   = ResourceMinLODClamp;
                 return desc;
             }),
             "MostDetailedMip"_a = 0, "MipLevels"_a = -1, "FirstArraySlice"_a = 0, "ArraySize"_a = 1, "PlaneSlice"_a = 0, "ResourceMinLODClamp"_a = 0.0f) //
        ;
    py::class_<D3D12_TEX2DMS_SRV>(m, "D3D12_TEX2DMS_SRV") //
        .def(py::init())                                  //
        ;
    py::class_<D3D12_TEX2DMS_ARRAY_SRV>(m, "D3D12_TEX2DMS_ARRAY_SRV") //
        .def(py::init([](UINT FirstArraySlice, UINT ArraySize) {
                 D3D12_TEX2DMS_ARRAY_SRV desc = {};
                 desc.FirstArraySlice         = FirstArraySlice;
                 desc.ArraySize               = ArraySize;
                 return desc;
             }),
             "FirstArraySlice"_a = 0, "ArraySize"_a = 1) //
        ;
    py::class_<D3D12_TEX3D_SRV>(m, "D3D12_TEX3D_SRV") //
        .def(py::init([](UINT MostDetailedMip, UINT MipLevels, FLOAT ResourceMinLODClamp) {
                 D3D12_TEX3D_SRV desc     = {};
                 desc.MostDetailedMip     = MostDetailedMip;
                 desc.MipLevels           = MipLevels;
                 desc.ResourceMinLODClamp = ResourceMinLODClamp;
                 return desc;
             }),
             "MostDetailedMip"_a = 0, "MipLevels"_a = -1, "ResourceMinLODClamp"_a = 0.0f) //
        ;
    py::class_<D3D12_TEXCUBE_SRV>(m, "D3D12_TEXCUBE_SRV") //
        .def(py::init([](UINT MostDetailedMip, UINT MipLevels, FLOAT ResourceMinLODClamp) {
                 D3D12_TEXCUBE_SRV desc   = {};
                 desc.MostDetailedMip     = MostDetailedMip;
                 desc.MipLevels           = MipLevels;
                 desc.ResourceMinLODClamp = ResourceMinLODClamp;
                 return desc;
             }),
             "MostDetailedMip"_a = 0, "MipLevels"_a = -1, "ResourceMinLODClamp"_a = 0.0f) //
        ;
    py::class_<D3D12_TEXCUBE_ARRAY_SRV>(m, "D3D12_TEXCUBE_ARRAY_SRV") //
        .def(py::init([](UINT MostDetailedMip, UINT MipLevels, UINT First2DArrayFace, UINT NumCubes, FLOAT ResourceMinLODClamp) {
                 D3D12_TEXCUBE_ARRAY_SRV desc = {};
                 desc.MostDetailedMip         = MostDetailedMip;
                 desc.MipLevels               = MipLevels;
                 desc.First2DArrayFace        = First2DArrayFace;
                 desc.NumCubes                = NumCubes;
                 desc.ResourceMinLODClamp     = ResourceMinLODClamp;
                 return desc;
             }),
             "MostDetailedMip"_a = 0, "MipLevels"_a = -1, "First2DArrayFace"_a = 0, "NumCubes"_a = 1, "ResourceMinLODClamp"_a = 0.0f) //
        ;
    py::class_<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV>(m, "D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV") //
        .def(py::init([](D3D12_GPU_VIRTUAL_ADDRESS Location) {
                 D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV desc = {};
                 desc.Location                                    = Location;
                 return desc;
             }),
             "Location"_a = 0) //
        ;

    m.attr("D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING") = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

    py::enum_<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS>(m, "D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS", py::arithmetic()) //
        .value("NONE", D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE)                                                                //
        .value("ALLOW_UPDATE", D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE)                                                //
        .value("ALLOW_COMPACTION", D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION)                                        //
        .value("PREFER_FAST_TRACE", D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE)                                      //
        .value("PREFER_FAST_BUILD", D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD)                                      //
        .value("MINIMIZE_MEMORY", D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY)                                          //
        .value("PERFORM_UPDATE", D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE)                                            //
        .export_values();

    py::enum_<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TYPE>(m, "D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TYPE") //
        .value("INFO_COMPACTED_SIZE", D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE)                                 //
        .value("INFO_TOOLS_VISUALIZATION", D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TOOLS_VISUALIZATION)                       //
        .export_values();

    py::class_<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC>(m, "D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC") //
        .def(py::init([](D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TYPE InfoType, D3D12_GPU_VIRTUAL_ADDRESS DestBuffer) {
                 D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC desc = {};
                 desc.InfoType                                                    = InfoType;
                 desc.DestBuffer                                                  = DestBuffer;
                 return desc;
             }),
             "InfoType"_a = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE, "DestBuffer"_a = 0) //
        ;

    py::enum_<D3D12_RAYTRACING_GEOMETRY_TYPE>(m, "D3D12_RAYTRACING_GEOMETRY_TYPE", py::arithmetic())    //
        .value("TRIANGLES", D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES)                                   //
        .value("PROCEDURAL_PRIMITIVE_AABBS", D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS) //
        .export_values();

    py::class_<D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE>(m, "D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE") //
        .def(py::init([](UINT64 StartAddress, UINT64 StrideInBytes, UINT64 SizeInBytes) {
                 D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE desc = {};
                 desc.StartAddress                               = StartAddress;
                 desc.StrideInBytes                              = StrideInBytes;
                 desc.SizeInBytes                                = SizeInBytes;
                 return desc;
             }),
             "StartAddress"_a = 0, "StrideInBytes"_a = 0, "SizeInBytes"_a = 0)                      //
        .def_readwrite("StartAddress", &D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE::StartAddress)   //
        .def_readwrite("StrideInBytes", &D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE::StrideInBytes) //
        .def_readwrite("SizeInBytes", &D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE::SizeInBytes)     //
        ;
    py::class_<D3D12_GPU_VIRTUAL_ADDRESS_RANGE>(m, "D3D12_GPU_VIRTUAL_ADDRESS_RANGE")  //
        .def(py::init())                                                               //
        .def_readwrite("StartAddress", &D3D12_GPU_VIRTUAL_ADDRESS_RANGE::StartAddress) //
        .def_readwrite("SizeInBytes", &D3D12_GPU_VIRTUAL_ADDRESS_RANGE::SizeInBytes)   //
        ;
    py::enum_<D3D12_RAYTRACING_INSTANCE_FLAGS>(m, "D3D12_RAYTRACING_INSTANCE_FLAGS", py::arithmetic())            //
        .value("NONE", D3D12_RAYTRACING_INSTANCE_FLAG_NONE)                                                       //
        .value("TRIANGLE_CULL_DISABLE", D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE)                     //
        .value("TRIANGLE_FRONT_COUNTERCLOCKWISE", D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE) //
        .value("FORCE_OPAQUE", D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE)                                       //
        .value("FORCE_NON_OPAQUE", D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_NON_OPAQUE)                               //
        .export_values();
    py::enum_<D3D12_RAYTRACING_GEOMETRY_FLAGS>(m, "D3D12_RAYTRACING_GEOMETRY_FLAGS", py::arithmetic())          //
        .value("NONE", D3D12_RAYTRACING_GEOMETRY_FLAG_NONE)                                                     //
        .value("OPAQUE", D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE)                                                 //
        .value("NO_DUPLICATE_ANYHIT_INVOCATION", D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION) //
        .export_values();
    py::class_<D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE>(m, "D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE") //
        .def(py::init([](D3D12_GPU_VIRTUAL_ADDRESS StartAddress, UINT64 StrideInBytes) {
                 D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE desc = {};
                 desc.StartAddress                         = StartAddress;
                 desc.StrideInBytes                        = StrideInBytes;
                 return desc;
             }),
             "StartAddress"_a = 0, "StrideInBytes"_a = 0) //
        ;
    py::class_<D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC>(m, "D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC") //
        .def(py::init([](UINT32 VertexCount, D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE VertexBuffer, DXGI_FORMAT VertexFormat, UINT IndexCount, D3D12_GPU_VIRTUAL_ADDRESS IndexBuffer,
                         DXGI_FORMAT IndexFormat, D3D12_GPU_VIRTUAL_ADDRESS Transform3x4) {
                 D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC desc = {};
                 desc.VertexCount                              = VertexCount;
                 desc.VertexBuffer                             = VertexBuffer;
                 desc.VertexFormat                             = VertexFormat;
                 desc.IndexCount                               = IndexCount;
                 desc.IndexBuffer                              = IndexBuffer;
                 desc.IndexFormat                              = IndexFormat;
                 desc.Transform3x4                             = Transform3x4;
                 return desc;
             }),
             "VertexCount"_a = 0, "VertexBuffer"_a = D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE{}, "VertexFormat"_a = DXGI_FORMAT_UNKNOWN, "IndexCount"_a = 0, "IndexBuffer"_a = 0,
             "IndexFormat"_a = DXGI_FORMAT_UNKNOWN, "Transform3x4"_a = D3D12_GPU_VIRTUAL_ADDRESS(0)) //

        ;

    py::class_<D3D12_RAYTRACING_GEOMETRY_AABBS_DESC>(m, "D3D12_RAYTRACING_GEOMETRY_AABBS_DESC") //
        .def(py::init([](UINT64 AABBCount, D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE AABBs) {
                 D3D12_RAYTRACING_GEOMETRY_AABBS_DESC desc = {};
                 desc.AABBCount                            = AABBCount;
                 desc.AABBs                                = AABBs;
                 return desc;
             }),
             "AABBCount"_a = 0, "AABBs"_a = D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE{}) //
        ;
    py::class_<D3D12_RAYTRACING_GEOMETRY_DESC>(m, "D3D12_RAYTRACING_GEOMETRY_DESC") //
        .def(py::init([](D3D12_RAYTRACING_GEOMETRY_FLAGS Flags, D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC Triangles) {
                 D3D12_RAYTRACING_GEOMETRY_DESC desc = {};
                 desc.Type                           = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
                 desc.Flags                          = Flags;
                 desc.Triangles                      = Triangles;
                 return desc;
             }),
             "Flags"_a = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE, "Triangles"_a = D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC{}) //
        .def(py::init([](D3D12_RAYTRACING_GEOMETRY_FLAGS Flags, D3D12_RAYTRACING_GEOMETRY_AABBS_DESC AABBs) {
                 D3D12_RAYTRACING_GEOMETRY_DESC desc = {};
                 desc.Type                           = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
                 desc.Flags                          = Flags;
                 desc.AABBs                          = AABBs;
                 return desc;
             }),
             "Flags"_a = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE, "AABBs"_a = D3D12_RAYTRACING_GEOMETRY_AABBS_DESC{}) //
        ;

    // D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS
    py::class_<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_Wrapper, std::shared_ptr<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_Wrapper>>(
        m, "D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS") //
        .def(py::init([](uint32_t Flags, UINT NumDescs, D3D12_GPU_VIRTUAL_ADDRESS InstanceDescs) {
                 return std::make_shared<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_Wrapper>((D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)Flags, NumDescs,
                                                                                                       InstanceDescs);
             }),
             "Flags"_a = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE, "NumDescs"_a = 0, "InstanceDescs"_a = 0) //
        .def(py::init([](uint32_t Flags, std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> GeometryDescs) {
                 return std::make_shared<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_Wrapper>((D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)Flags, GeometryDescs);
             }),
             "Flags"_a = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE, "GeometryDescs"_a = std::vector<D3D12_RAYTRACING_GEOMETRY_DESC>{}) //
        ;

    // D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO
    py::class_<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO>(m, "D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO")            //
        .def_readwrite("ResultDataMaxSizeInBytes", &D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::ResultDataMaxSizeInBytes)         //
        .def_readwrite("ScratchDataSizeInBytes", &D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::ScratchDataSizeInBytes)             //
        .def_readwrite("UpdateScratchDataSizeInBytes", &D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::UpdateScratchDataSizeInBytes) //
        ;

    py::class_<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC_WRAPPER, std::shared_ptr<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC_WRAPPER>>(
        m, "D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC") //
        .def(py::init([](D3D12_GPU_VIRTUAL_ADDRESS Source, D3D12_GPU_VIRTUAL_ADDRESS Dest, D3D12_GPU_VIRTUAL_ADDRESS Scratch,
                         std::shared_ptr<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_Wrapper> Inputs) {
                 return std::make_shared<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC_WRAPPER>(Inputs, Source, Dest, Scratch);
             }),
             "Source"_a = 0, "Dest"_a = 0, "Scratch"_a = 0, "Inputs"_a = nullptr) //
        ;

    py::enum_<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE>(m, "D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE") //
        .value("TOP_LEVEL", D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL)                            //
        .value("BOTTOM_LEVEL", D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL)                      //
        .export_values();

    py::enum_<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE>(m, "D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE") //
        .value("CLONE", D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE)                                         //
        .value("COMPACT", D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT)                                     //
        .export_values();

    py::class_<D3D12_SHADER_RESOURCE_VIEW_DESC>(m, "D3D12_SHADER_RESOURCE_VIEW_DESC") //
        .def(py::init([](DXGI_FORMAT Format, UINT Shader4ComponentMapping, D3D12_BUFFER_SRV Buffer) {
                 D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
                 desc.Format                          = Format;
                 desc.Shader4ComponentMapping         = Shader4ComponentMapping;
                 desc.ViewDimension                   = D3D12_SRV_DIMENSION_BUFFER;
                 desc.Buffer                          = Buffer;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Shader4ComponentMapping"_a = 0, "Buffer"_a = D3D12_BUFFER_SRV{}) //
        .def(py::init([](DXGI_FORMAT Format, UINT Shader4ComponentMapping, D3D12_TEX1D_SRV Texture1D) {
                 D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
                 desc.Format                          = Format;
                 desc.Shader4ComponentMapping         = Shader4ComponentMapping;
                 desc.ViewDimension                   = D3D12_SRV_DIMENSION_TEXTURE1D;
                 desc.Texture1D                       = Texture1D;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Shader4ComponentMapping"_a = 0, "Texture1D"_a = D3D12_TEX1D_SRV{}) //
        .def(py::init([](DXGI_FORMAT Format, UINT Shader4ComponentMapping, D3D12_TEX1D_ARRAY_SRV Texture1DArray) {
                 D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
                 desc.Format                          = Format;
                 desc.Shader4ComponentMapping         = Shader4ComponentMapping;
                 desc.ViewDimension                   = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
                 desc.Texture1DArray                  = Texture1DArray;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Shader4ComponentMapping"_a = 0, "Texture1DArray"_a = D3D12_TEX1D_ARRAY_SRV{}) //
        .def(py::init([](DXGI_FORMAT Format, UINT Shader4ComponentMapping, D3D12_TEX2D_SRV Texture2D) {
                 D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
                 desc.Format                          = Format;
                 desc.Shader4ComponentMapping         = Shader4ComponentMapping;
                 desc.ViewDimension                   = D3D12_SRV_DIMENSION_TEXTURE2D;
                 desc.Texture2D                       = Texture2D;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Shader4ComponentMapping"_a = 0, "Texture2D"_a = D3D12_TEX2D_SRV{}) //
        .def(py::init([](DXGI_FORMAT Format, UINT Shader4ComponentMapping, D3D12_TEX2D_ARRAY_SRV Texture2DArray) {
                 D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
                 desc.Format                          = Format;
                 desc.Shader4ComponentMapping         = Shader4ComponentMapping;
                 desc.ViewDimension                   = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                 desc.Texture2DArray                  = Texture2DArray;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Shader4ComponentMapping"_a = 0, "Texture2DArray"_a = D3D12_TEX2D_ARRAY_SRV{}) //
        .def(py::init([](DXGI_FORMAT Format, UINT Shader4ComponentMapping, D3D12_TEX2DMS_SRV Texture2DMS) {
                 D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
                 desc.Format                          = Format;
                 desc.Shader4ComponentMapping         = Shader4ComponentMapping;
                 desc.ViewDimension                   = D3D12_SRV_DIMENSION_TEXTURE2DMS;
                 desc.Texture2DMS                     = Texture2DMS;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Shader4ComponentMapping"_a = 0, "Texture2DMS"_a = D3D12_TEX2DMS_SRV{}) //
        .def(py::init([](DXGI_FORMAT Format, UINT Shader4ComponentMapping, D3D12_TEX2DMS_ARRAY_SRV Texture2DMSArray) {
                 D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
                 desc.Format                          = Format;
                 desc.Shader4ComponentMapping         = Shader4ComponentMapping;
                 desc.ViewDimension                   = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
                 desc.Texture2DMSArray                = Texture2DMSArray;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Shader4ComponentMapping"_a = 0, "Texture2DMSArray"_a = D3D12_TEX2DMS_ARRAY_SRV{}) //
        .def(py::init([](DXGI_FORMAT Format, UINT Shader4ComponentMapping, D3D12_TEX3D_SRV Texture3D) {
                 D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
                 desc.Format                          = Format;
                 desc.Shader4ComponentMapping         = Shader4ComponentMapping;
                 desc.ViewDimension                   = D3D12_SRV_DIMENSION_TEXTURE3D;
                 desc.Texture3D                       = Texture3D;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Shader4ComponentMapping"_a = 0, "Texture3D"_a = D3D12_TEX3D_SRV{}) //
        .def(py::init([](DXGI_FORMAT Format, UINT Shader4ComponentMapping, D3D12_TEXCUBE_SRV TextureCube) {
                 D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
                 desc.Format                          = Format;
                 desc.Shader4ComponentMapping         = Shader4ComponentMapping;
                 desc.ViewDimension                   = D3D12_SRV_DIMENSION_TEXTURECUBE;
                 desc.TextureCube                     = TextureCube;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Shader4ComponentMapping"_a = 0, "TextureCube"_a = D3D12_TEXCUBE_SRV{}) //
        .def(py::init([](DXGI_FORMAT Format, UINT Shader4ComponentMapping, D3D12_TEXCUBE_ARRAY_SRV TextureCubeArray) {
                 D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
                 desc.Format                          = Format;
                 desc.Shader4ComponentMapping         = Shader4ComponentMapping;
                 desc.ViewDimension                   = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
                 desc.TextureCubeArray                = TextureCubeArray;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Shader4ComponentMapping"_a = 0, "TextureCubeArray"_a = D3D12_TEXCUBE_ARRAY_SRV{}) //
        .def(py::init([](DXGI_FORMAT Format, UINT Shader4ComponentMapping, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV RaytracingAccelerationStructure) {
                 D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
                 desc.Format                          = Format;
                 desc.Shader4ComponentMapping         = Shader4ComponentMapping;
                 desc.ViewDimension                   = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
                 desc.RaytracingAccelerationStructure = RaytracingAccelerationStructure;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Shader4ComponentMapping"_a = 0, "RaytracingAccelerationStructure"_a = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV{}) //
        ;
    py::enum_<D3D12_BUFFER_UAV_FLAGS>(m, "D3D12_BUFFER_UAV_FLAGS", py::arithmetic()) //
        .value("NONE", D3D12_BUFFER_UAV_FLAG_NONE)                                   //
        .value("RAW", D3D12_BUFFER_UAV_FLAG_RAW)                                     //
        .export_values();
    py::class_<D3D12_BUFFER_UAV>(m, "D3D12_BUFFER_UAV") //
        .def(py::init([](UINT64 FirstElement, UINT NumElements, UINT StructureByteStride, UINT64 CounterOffsetInBytes, D3D12_BUFFER_UAV_FLAGS Flags) {
                 D3D12_BUFFER_UAV desc     = {};
                 desc.FirstElement         = FirstElement;
                 desc.NumElements          = NumElements;
                 desc.StructureByteStride  = StructureByteStride;
                 desc.CounterOffsetInBytes = CounterOffsetInBytes;
                 desc.Flags                = Flags;
                 return desc;
             }),
             "FirstElement"_a = 0, "NumElements"_a = 1, "StructureByteStride"_a = 0, "CounterOffsetInBytes"_a = 0, "Flags"_a = D3D12_BUFFER_UAV_FLAG_NONE) //
        ;
    py::class_<D3D12_TEX1D_UAV>(m, "D3D12_TEX1D_UAV") //
        .def(py::init([](UINT MipSlice) {
                 D3D12_TEX1D_UAV desc = {};
                 desc.MipSlice        = MipSlice;
                 return desc;
             }),
             "MipSlice"_a = 0) //
        ;
    py::class_<D3D12_TEX1D_ARRAY_UAV>(m, "D3D12_TEX1D_ARRAY_UAV") //
        .def(py::init([](UINT MipSlice, UINT FirstArraySlice, UINT ArraySize) {
                 D3D12_TEX1D_ARRAY_UAV desc = {};
                 desc.MipSlice              = MipSlice;
                 desc.FirstArraySlice       = FirstArraySlice;
                 desc.ArraySize             = ArraySize;
                 return desc;
             }),
             "MipSlice"_a = 0, "FirstArraySlice"_a = 0, "ArraySize"_a = 1) //
        ;
    py::class_<D3D12_TEX2D_UAV>(m, "D3D12_TEX2D_UAV") //
        .def(py::init([](UINT MipSlice, UINT PlaneSlice) {
                 D3D12_TEX2D_UAV desc = {};
                 desc.MipSlice        = MipSlice;
                 desc.PlaneSlice      = PlaneSlice;
                 return desc;
             }),
             "MipSlice"_a = 0, "PlaneSlice"_a = 0) //
        ;
    py::class_<D3D12_TEX2D_ARRAY_UAV>(m, "D3D12_TEX2D_ARRAY_UAV") //
        .def(py::init([](UINT MipSlice, UINT FirstArraySlice, UINT ArraySize, UINT PlaneSlice) {
                 D3D12_TEX2D_ARRAY_UAV desc = {};
                 desc.MipSlice              = MipSlice;
                 desc.FirstArraySlice       = FirstArraySlice;
                 desc.ArraySize             = ArraySize;
                 desc.PlaneSlice            = PlaneSlice;
                 return desc;
             }),
             "MipSlice"_a = 0, "FirstArraySlice"_a = 0, "ArraySize"_a = 1, "PlaneSlice"_a = 0) //
        ;
    py::class_<D3D12_TEX3D_UAV>(m, "D3D12_TEX3D_UAV") //
        .def(py::init([](UINT MipSlice, UINT FirstWSlice, UINT WSize) {
                 D3D12_TEX3D_UAV desc = {};
                 desc.MipSlice        = MipSlice;
                 desc.FirstWSlice     = FirstWSlice;
                 desc.WSize           = WSize;
                 return desc;
             }),
             "MipSlice"_a = 0, "FirstWSlice"_a = 0, "WSize"_a = 1) //
        ;
    py::class_<D3D12_UNORDERED_ACCESS_VIEW_DESC>(m, "D3D12_UNORDERED_ACCESS_VIEW_DESC") //
        .def(py::init([](DXGI_FORMAT Format, D3D12_BUFFER_UAV Buffer) {
                 D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
                 desc.Format                           = Format;
                 desc.ViewDimension                    = D3D12_UAV_DIMENSION_BUFFER;
                 desc.Buffer                           = Buffer;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Buffer"_a = D3D12_BUFFER_UAV{}) //
        .def(py::init([](DXGI_FORMAT Format, D3D12_TEX1D_UAV Texture1D) {
                 D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
                 desc.Format                           = Format;
                 desc.ViewDimension                    = D3D12_UAV_DIMENSION_TEXTURE1D;
                 desc.Texture1D                        = Texture1D;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Texture1D"_a = D3D12_TEX1D_UAV{}) //
        .def(py::init([](DXGI_FORMAT Format, D3D12_TEX1D_ARRAY_UAV Texture1DArray) {
                 D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
                 desc.Format                           = Format;
                 desc.ViewDimension                    = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
                 desc.Texture1DArray                   = Texture1DArray;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Texture1DArray"_a = D3D12_TEX1D_ARRAY_UAV{}) //
        .def(py::init([](DXGI_FORMAT Format, D3D12_TEX2D_UAV Texture2D) {
                 D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
                 desc.Format                           = Format;
                 desc.ViewDimension                    = D3D12_UAV_DIMENSION_TEXTURE2D;
                 desc.Texture2D                        = Texture2D;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Texture2D"_a = D3D12_TEX2D_UAV{}) //
        .def(py::init([](DXGI_FORMAT Format, D3D12_TEX2D_ARRAY_UAV Texture2DArray) {
                 D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
                 desc.Format                           = Format;
                 desc.ViewDimension                    = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                 desc.Texture2DArray                   = Texture2DArray;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Texture2DArray"_a = D3D12_TEX2D_ARRAY_UAV{}) //
        .def(py::init([](DXGI_FORMAT Format, D3D12_TEX3D_UAV Texture3D) {
                 D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
                 desc.Format                           = Format;
                 desc.ViewDimension                    = D3D12_UAV_DIMENSION_TEXTURE3D;
                 desc.Texture3D                        = Texture3D;
                 return desc;
             }),
             "Format"_a = DXGI_FORMAT_UNKNOWN, "Texture3D"_a = D3D12_TEX3D_UAV{}) //
        ;

    py::class_<ID3D12DeviceWrapper, std::shared_ptr<ID3D12DeviceWrapper>>(m, "ID3D12Device")                                                                //
        .def(py::init<ID3D12Device *>())                                                                                                                    //
        .def("GetCopyableFootprints", &ID3D12DeviceWrapper::GetCopyableFootprints,                                                                          //
             "Resource"_a, "FirstSubresource"_a, "NumSubresources"_a, "BaseOffset"_a)                                                                       //
        .def("CreateCommittedResource", &ID3D12DeviceWrapper::CreateCommittedResource,                                                                      //
             "heapProperties"_a,                                                                                                                            //
             "heapFlags"_a,                                                                                                                                 //
             "resourceDesc"_a,                                                                                                                              //
             "initialState"_a,                                                                                                                              //
             "optimizedClearValue"_a = std::optional<D3D12_CLEAR_VALUE>{}                                                                                   //
             )                                                                                                                                              //
        .def("CreateRootSignature", &ID3D12DeviceWrapper::CreateRootSignature, "NodeMask"_a = 0, "Bytes"_a)                                                 //
        .def("CreateComputePipelineState", &ID3D12DeviceWrapper::CreateComputePipelineState)                                                                //
        .def("CreateCommandQueue", &ID3D12DeviceWrapper::CreateCommandQueue, "Desc"_a)                                                                      //
        .def("CreateFence", &ID3D12DeviceWrapper::CreateFence, "InitialValue"_a = 0, "Flags"_a = D3D12_FENCE_FLAG_NONE)                                     //
        .def("CreateCommandAllocator", &ID3D12DeviceWrapper::CreateCommandAllocator, "Type"_a)                                                              //
        .def("CreateCommandList", &ID3D12DeviceWrapper::CreateCommandList, "NodeMask"_a, "Type"_a, "Allocator"_a, "InitialPSO"_a = nullptr)                 //
        .def("CreateDescriptorHeap", &ID3D12DeviceWrapper::CreateDescriptorHeap, "Desc"_a)                                                                  //
        .def("CreateShaderResourceView", &ID3D12DeviceWrapper::CreateShaderResourceView, "Resource"_a, "Desc"_a, "DestDescriptor"_a)                        //
        .def("CreateGraphicsPipelineState", &ID3D12DeviceWrapper::CreateGraphicsPipelineState, "Desc"_a)                                                    //
        .def("CreateRenderTargetView", &ID3D12DeviceWrapper::CreateRenderTargetView, "Resource"_a, "Desc"_a, "DestDescriptor"_a)                            //
        .def("CreateDepthStencilView", &ID3D12DeviceWrapper::CreateDepthStencilView, "Resource"_a, "Desc"_a, "DestDescriptor"_a)                            //
        .def("CreateConstantBufferView", &ID3D12DeviceWrapper::CreateConstantBufferView, "Desc"_a, "DestDescriptor"_a)                                      //
        .def("CreateUnorderedAccessView", &ID3D12DeviceWrapper::CreateUnorderedAccessView, "Resource"_a, "CounterResource"_a, "Desc"_a, "DestDescriptor"_a) //
        .def("CreateSampler", &ID3D12DeviceWrapper::CreateSampler, "Desc"_a, "DestDescriptor"_a)                                                            //
        .def("GetDescriptorHandleIncrementSize", &ID3D12DeviceWrapper::GetDescriptorHandleIncrementSize, "Type"_a)                                          //
        .def("GetRaytracingAccelerationStructurePrebuildInfo", &ID3D12DeviceWrapper::GetRaytracingAccelerationStructurePrebuildInfo, "Inputs"_a)            //
        ;

    m.def("CreateDevice", &CreateDevice);

    py::class_<ID3D12DebugWrapper, std::shared_ptr<ID3D12DebugWrapper>>(m, "ID3D12Debug")     //
        .def(py::init())                                                                      //
        .def("EnableDebugLayer", &ID3D12DebugWrapper::EnableDebugLayer)                       //
        .def("SetEnableGPUBasedValidation", &ID3D12DebugWrapper::SetEnableGPUBasedValidation) //
        ;
}
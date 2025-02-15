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

#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <d3dx12/d3dx12.h>
#include <dxgi1_6.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

float square(float x) { return x * x; }

#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)

#define ASSERT_HRESULT_PANIC(hr)                                                                                                                                                   \
    if (FAILED(hr)) {                                                                                                                                                              \
        throw std::runtime_error("" __FILE__ ":" STRINGIFY(__LINE__) " " + std::to_string(hr));                                                                                    \
    }

#define ASSERT_PANIC(cond)                                                                                                                                                         \
    if (!(cond)) {                                                                                                                                                                 \
        throw std::runtime_error("" __FILE__ ":" STRINGIFY(__LINE__) " " #cond);                                                                                                   \
    }

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

    void Signal(UINT64 value) { fence->Signal(value); }
    void GetCompletedValue() { fence->GetCompletedValue(); }
    void SetEventOnCompletion(UINT64 value, HANDLE hEvent) { fence->SetEventOnCompletion(value, hEvent); }
};
class ID3D12PipelineStateWrapper {
public:
    ID3D12PipelineState *pipelineState = nullptr;

public:
    ID3D12PipelineStateWrapper(ID3D12PipelineState *pipelineState) : pipelineState(pipelineState) {}

    ~ID3D12PipelineStateWrapper() {
        if (pipelineState) pipelineState->Release();
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
    void ClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE RenderTargetView, std::vector<float> ColorRGBA, std::vector<D3D12_RECT> Rects) {
        UINT NumRects = (UINT)Rects.size();
        ASSERT_PANIC(NumRects <= 8);
        ASSERT_PANIC(ColorRGBA.size() == 4);
        commandList->ClearRenderTargetView(RenderTargetView, ColorRGBA.data(), NumRects, Rects.data());
    }
    void ClearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE DepthStencilView, D3D12_CLEAR_FLAGS ClearFlags, FLOAT Depth, UINT8 Stencil, std::vector<D3D12_RECT> Rects) {
        UINT NumRects = (UINT)Rects.size();
        commandList->ClearDepthStencilView(DepthStencilView, ClearFlags, Depth, Stencil, NumRects, Rects.data());
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
    std::shared_ptr<IDXGISwapChainWrapper> CreateSwapChain(std::shared_ptr<ID3D12CommandQueueWrapper> commandQueue, DXGI_SWAP_CHAIN_DESC *desc) {
        IDXGISwapChain *swapChain = nullptr;
        HRESULT         hr        = factory->CreateSwapChain(commandQueue->commandQueue, desc, &swapChain);
        ASSERT_HRESULT_PANIC(hr);
        return std::make_shared<IDXGISwapChainWrapper>(swapChain);
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

class D3D12_GRAPHICS_PIPELINE_STATE_DESC_Wrapper {
public:
    std::shared_ptr<ID3D12RootSignatureWrapper>    pRootSignature        = {};
    std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper> VS                    = {};
    std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper> PS                    = {};
    std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper> DS                    = {};
    std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper> HS                    = {};
    std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper> GS                    = {};
    D3D12_STREAM_OUTPUT_DESC                       StreamOutput          = {};
    D3D12_BLEND_DESC                               BlendState            = {};
    UINT                                           SampleMask            = {};
    D3D12_RASTERIZER_DESC                          RasterizerState       = {};
    D3D12_DEPTH_STENCIL_DESC                       DepthStencilState     = {};
    D3D12_INPUT_LAYOUT_DESC                        InputLayout           = {};
    D3D12_INDEX_BUFFER_STRIP_CUT_VALUE             IBStripCutValue       = {};
    D3D12_PRIMITIVE_TOPOLOGY_TYPE                  PrimitiveTopologyType = {};
    UINT                                           NumRenderTargets      = {};
    std::array<DXGI_FORMAT, 8>                     RTVFormats            = {};
    DXGI_FORMAT                                    DSVFormat             = {};
    DXGI_SAMPLE_DESC                               SampleDesc            = {};
    UINT                                           NodeMask              = {};
    D3D12_CACHED_PIPELINE_STATE                    CachedPSO             = {};
    D3D12_PIPELINE_STATE_FLAGS                     Flags                 = {};

public:
    D3D12_GRAPHICS_PIPELINE_STATE_DESC_Wrapper(std::shared_ptr<ID3D12RootSignatureWrapper>    pRootSignature = {}, //
                                               std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper> VS             = {}, //
                                               std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper> PS             = {}, //
                                               std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper> DS             = {}, //
                                               std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper> HS             = {}, //
                                               std::shared_ptr<D3D12_SHADER_BYTECODE_Wrapper> GS             = {}, //
                                               D3D12_STREAM_OUTPUT_DESC StreamOutput = {}, D3D12_BLEND_DESC BlendState = {}, UINT SampleMask = {},
                                               D3D12_RASTERIZER_DESC RasterizerState = {}, D3D12_DEPTH_STENCIL_DESC DepthStencilState = {},
                                               D3D12_INPUT_LAYOUT_DESC InputLayout = {}, D3D12_INDEX_BUFFER_STRIP_CUT_VALUE IBStripCutValue = {},
                                               D3D12_PRIMITIVE_TOPOLOGY_TYPE PrimitiveTopologyType = {}, UINT NumRenderTargets = {}, std::array<DXGI_FORMAT, 8> RTVFormats = {},
                                               DXGI_FORMAT DSVFormat = {}, DXGI_SAMPLE_DESC SampleDesc = {}, UINT NodeMask = {}, D3D12_CACHED_PIPELINE_STATE CachedPSO = {},
                                               D3D12_PIPELINE_STATE_FLAGS Flags = {})
        : pRootSignature(pRootSignature), VS(VS), PS(PS), DS(DS), HS(HS), GS(GS), StreamOutput(StreamOutput), BlendState(BlendState), SampleMask(SampleMask),
          RasterizerState(RasterizerState), DepthStencilState(DepthStencilState), InputLayout(InputLayout), IBStripCutValue(IBStripCutValue),
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
        desc.InputLayout                        = InputLayout;
        desc.IBStripCutValue                    = IBStripCutValue;
        desc.PrimitiveTopologyType              = PrimitiveTopologyType;
        desc.NumRenderTargets                   = NumRenderTargets;
        for (UINT i = 0; i < 8; i++) {
            desc.RTVFormats[i] = RTVFormats[i];
        }
        desc.DSVFormat  = DSVFormat;
        desc.SampleDesc = SampleDesc;
        desc.NodeMask   = NodeMask;
        desc.CachedPSO  = CachedPSO;
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
    std::shared_ptr<ID3D12PipelineStateWrapper> CreateComputePipelineState(std::shared_ptr<D3D12_COMPUTE_PIPELINE_STATE_DESC_Wrapper> desc) {
        ID3D12PipelineState *pipelineState = nullptr;
        HRESULT              hr            = device->CreateComputePipelineState(&desc->ToNative(), IID_PPV_ARGS(&pipelineState));
        ASSERT_HRESULT_PANIC(hr);
        return std::make_shared<ID3D12PipelineStateWrapper>(pipelineState);
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

PYBIND11_MODULE(native, m) {
    m.def("square", &square);

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

    py::class_<DXGI_SWAP_CHAIN_DESC>(m, "DXGI_SWAP_CHAIN_DESC") //
        .def(py::init())                                        //
        .def(py::init([](UINT width, UINT height, DXGI_FORMAT format, UINT bufferCount, uint64_t outputWindow, BOOL windowed, DXGI_SWAP_EFFECT swapEffect, UINT flags) {
                 DXGI_SWAP_CHAIN_DESC desc = {};
                 desc.BufferDesc.Width     = width;
                 desc.BufferDesc.Height    = height;
                 desc.BufferDesc.Format    = format;
                 desc.BufferCount          = bufferCount;
                 desc.OutputWindow         = (HWND)outputWindow;
                 desc.Windowed             = windowed;
                 desc.SwapEffect           = swapEffect;
                 desc.Flags                = flags;
                 return desc;
             }),                                                                                                            //
             "Width"_a, "Height"_a, "Format"_a, "BufferCount"_a, "OutputWindow"_a, "Windowed"_a, "SwapEffect"_a, "Flags"_a) //

        .def_readwrite("BufferDesc", &DXGI_SWAP_CHAIN_DESC::BufferDesc)     //
        .def_readwrite("SampleDesc", &DXGI_SWAP_CHAIN_DESC::SampleDesc)     //
        .def_readwrite("BufferUsage", &DXGI_SWAP_CHAIN_DESC::BufferUsage)   //
        .def_readwrite("BufferCount", &DXGI_SWAP_CHAIN_DESC::BufferCount)   //
        .def_readwrite("OutputWindow", &DXGI_SWAP_CHAIN_DESC::OutputWindow) //
        .def_readwrite("Windowed", &DXGI_SWAP_CHAIN_DESC::Windowed)         //
        .def_readwrite("SwapEffect", &DXGI_SWAP_CHAIN_DESC::SwapEffect)     //
        .def_readwrite("Flags", &DXGI_SWAP_CHAIN_DESC::Flags)               //
        ;

    py::class_<IDXGIAdapterWrapper, std::shared_ptr<IDXGIAdapterWrapper>>(m, "IDXGIAdapter") //
        .def(py::init<IDXGIAdapter *>())                                                     //
        .def("GetDesc", &IDXGIAdapterWrapper::GetDesc)                                       //
        ;

    py::class_<IDXGIFactoryWrapper, std::shared_ptr<IDXGIFactoryWrapper>>(m, "IDXGIFactory") //
        .def(py::init())                                                                     //
        .def("GetAdapter", &IDXGIFactoryWrapper::GetAdapter)                                 //
        .def("EnumAdapters", &IDXGIFactoryWrapper::EnumAdapters)                             //
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
             "pRootSignature"_a = nullptr, "CS"_a = nullptr, "NodeMask"_a = 0, "CachedPSO"_a = std::nullopt, "Flags"_a = D3D12_PIPELINE_STATE_FLAG_NONE) //
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

    py::class_<DXGI_SAMPLE_DESC>(m, "DXGI_SAMPLE_DESC") //
        .def(py::init())                                //
        .def(py::init([](UINT Count, UINT Quality) {
                 DXGI_SAMPLE_DESC desc = {};
                 desc.Count            = Count;
                 desc.Quality          = Quality;
                 return desc;
             }),
             "Count"_a = 1, "Quality"_a = 0)                  //
        .def_readwrite("Count", &DXGI_SAMPLE_DESC::Count)     //
        .def_readwrite("Quality", &DXGI_SAMPLE_DESC::Quality) //
        ;

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
        ;

    py::class_<ID3D12PipelineStateWrapper, std::shared_ptr<ID3D12PipelineStateWrapper>>(m, "ID3D12PipelineState") //
        ;
    py::class_<ID3D12DeviceWrapper, std::shared_ptr<ID3D12DeviceWrapper>>(m, "ID3D12Device")                //
        .def(py::init<ID3D12Device *>())                                                                    //
        .def("CreateCommittedResource", &ID3D12DeviceWrapper::CreateCommittedResource,                      //
             "heapProperties"_a,                                                                            //
             "heapFlags"_a,                                                                                 //
             "resourceDesc"_a,                                                                              //
             "initialState"_a,                                                                              //
             "optimizedClearValue"_a = std::optional<D3D12_CLEAR_VALUE>{}                                   //
             )                                                                                              //
        .def("CreateRootSignature", &ID3D12DeviceWrapper::CreateRootSignature, "NodeMask"_a = 0, "Bytes"_a) //
        .def("CreateComputePipelineState", &ID3D12DeviceWrapper::CreateComputePipelineState)                //
        ;

    m.def("CreateDevice", &CreateDevice);
}
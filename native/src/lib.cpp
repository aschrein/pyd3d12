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

#include <d3dx12/d3dx12.h>
#include <dxgi1_6.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

float square(float x) { return x * x; }

#define STRINGIFY(x) #x

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
};

static std::shared_ptr<ID3D12DeviceWrapper> CreateDevice(std::shared_ptr<IDXGIAdapterWrapper> adapter, D3D_FEATURE_LEVEL featureLevel) {
    ID3D12Device *device = nullptr;
    HRESULT       hr     = D3D12CreateDevice(adapter->adapter, featureLevel, IID_PPV_ARGS(&device));
    ASSERT_HRESULT_PANIC(hr);
    return std::make_shared<ID3D12DeviceWrapper>(device);
}

PYBIND11_MODULE(native, m) {
    m.def("square", &square);

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

    py::class_<IDXGIFactoryWrapper, std::shared_ptr<IDXGIFactoryWrapper>>(m, "IDXGIFactory") //
        .def(py::init())                                                                     //
        .def("GetAdapter", &IDXGIFactoryWrapper::GetAdapter)                                 //
        .def("EnumAdapters", &IDXGIFactoryWrapper::EnumAdapters)                             //
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

    py::class_<ID3D12DeviceWrapper, std::shared_ptr<ID3D12DeviceWrapper>>(m, "ID3D12Device") //
        .def(py::init<ID3D12Device *>())                                                  //
        ;

    m.def("CreateDevice", &CreateDevice);

}
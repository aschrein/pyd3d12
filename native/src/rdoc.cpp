/*
#  MIT License
#  
#  Copyright (c) 2025 Anton Schreiner
#  
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
*/

#include "common.h"

#include <renderdoc_app.h>

#include <Windows.h>

class RDContext {
public:
    RENDERDOC_API_1_4_1 *ctx = nullptr;

public:
    RDContext() = default;
    RDContext(std::string dll_path) {
        // HMODULE mod = LoadLibraryA("renderdoc.dll");
        HMODULE mod = LoadLibraryA(dll_path.c_str());
        if (mod == nullptr) {
            throw std::runtime_error("Failed to load renderdoc.dll");
        }

        pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
        if (RENDERDOC_GetAPI == nullptr) {
            throw std::runtime_error("Failed to get RENDERDOC_GetAPI");
        }

        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_4_1, (void **)&ctx);
        if (ret != 1) {
            throw std::runtime_error("Failed to get RENDERDOC_API_1_4_1");
        }
    }

    void SetCaptureFilePathTemplate(std::string path) { ctx->SetCaptureFilePathTemplate(path.c_str()); }

    ~RDContext() {
        if (ctx) {
            ctx->Shutdown();
            ctx = nullptr;
        }
    }

    bool IsValid() { return ctx != nullptr; }
    bool IsCapturing() { return ctx->IsFrameCapturing() != 0; }
    void StartCapture() { ctx->StartFrameCapture(nullptr, nullptr); }
    void EndCapture() { ctx->EndFrameCapture(nullptr, nullptr); }
};

static std::shared_ptr<RDContext> CreateContext(std::string dll_path) { return std::make_shared<RDContext>(dll_path); }

PYBIND11_MODULE(rdoc, m) {

    py::class_<RDContext, std::shared_ptr<RDContext>>(m, "Context")
        .def(py::init<>())
        .def("IsValid", &RDContext::IsValid)
        .def("IsCapturing", &RDContext::IsCapturing)
        .def("StartCapture", &RDContext::StartCapture)
        .def("EndCapture", &RDContext::EndCapture) //
        .def("SetCaptureFilePathTemplate", &RDContext::SetCaptureFilePathTemplate);

    m.def("CreateContext", &CreateContext);
}
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

// #include "imgui.h"
#include "imgui_internal.h"

#include <Windows.h>

// class ImGuiContext;

// Non owning pointer
template <typename T> class ProxyPtr {
public:
    T *ptr;

public:
    ProxyPtr(T *ptr) : ptr(ptr) {}
    ProxyPtr() : ptr(nullptr) {}
       operator T *() { return ptr; }
    T *operator->() { return ptr; }
};

class ContextWrapper {
public:
    ImGuiContext *             ctx                  = nullptr;
    HWND                       hwnd                 = {};
    std::vector<unsigned char> font_pixels          = {};
    int                        font_width           = {};
    int                        font_height          = {};
    int                        font_bytes_per_pixel = {};

public:
    ContextWrapper() {}
    ContextWrapper(ImGuiContext *_ctx, uint64_t _hwnd) : ctx(_ctx), hwnd((HWND)_hwnd) {
        ImGui::SetCurrentContext(ctx);
        auto &io = ImGui::GetIO();
        io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;
        RECT rect;
        GetClientRect(hwnd, &rect);
        io.DisplaySize             = ImVec2((float)(rect.right - rect.left), (float)(rect.bottom - rect.top));
        io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
        io.MouseDrawCursor         = true;
        unsigned char *out_pixels  = {};

        io.Fonts->GetTexDataAsRGBA32(&out_pixels, &font_width, &font_height, &font_bytes_per_pixel);
        font_pixels.resize(font_width * font_height * font_bytes_per_pixel);
        ASSERT_PANIC(font_pixels.size() == font_width * font_height * font_bytes_per_pixel);
        ASSERT_PANIC(font_bytes_per_pixel == 4);
        memcpy(font_pixels.data(), out_pixels, font_pixels.size());
    }

    uint64_t GetFontTexturePtr() { return (uint64_t)font_pixels.data(); }
    int      GetFontTextureWidth() { return font_width; }
    int      GetFontTextureHeight() { return font_height; }
    int      GetFontTextureBytesPerPixel() { return font_bytes_per_pixel; }

                  operator ImGuiContext *() { return ctx; }
    ImGuiContext *operator->() { return ctx; }

    void Destroy() {
        if (ctx) {
            ImGui::DestroyContext(ctx);
            ctx = nullptr;
        }
        hwnd = {};
    }
};

void export_imgui_0(py::module &m) {

    py::class_<ImVec2>(m, "Vec2").def(py::init<float, float>()).def_readwrite("x", &ImVec2::x).def_readwrite("y", &ImVec2::y);

    py::class_<ImVec4>(m, "Vec4")
        .def(py::init<float, float, float, float>())
        .def_readwrite("x", &ImVec4::x)
        .def_readwrite("y", &ImVec4::y)
        .def_readwrite("z", &ImVec4::z)
        .def_readwrite("w", &ImVec4::w);

    py::class_<ImColor>(m, "Color")
        .def(py::init<>())
        .def(py::init<>([](float r, float g, float b, float a) { return ImColor(r, g, b, a); }), "r"_a, "g"_a, "b"_a, "a"_a)
        .def("setHSV", &ImColor::SetHSV)
        .def("setHSV", &ImColor::SetHSV, "h"_a, "s"_a, "v"_a, "a"_a)
        .def_readwrite("Value", &ImColor::Value);

    py::class_<ImDrawCmd>(m, "DrawCmd")
        .def_readwrite("ElemCount", &ImDrawCmd::ElemCount)
        .def_readwrite("ClipRect", &ImDrawCmd::ClipRect)
        .def_readwrite("TextureId", &ImDrawCmd::TextureId)
        .def_readwrite("VtxOffset", &ImDrawCmd::VtxOffset)
        .def_readwrite("IdxOffset", &ImDrawCmd::IdxOffset)
        .def_readwrite("UserCallback", &ImDrawCmd::UserCallback)
        .def_readwrite("UserCallbackData", &ImDrawCmd::UserCallbackData);

    py::class_<ImDrawList>(m, "DrawList")
        .def_property_readonly("CmdBuffer", [](ImDrawList &self) { return std::vector<ImDrawCmd>(self.CmdBuffer.Data, self.CmdBuffer.Data + self.CmdBuffer.Size); }) //
        .def_property_readonly("IdxBufferPtr", [](ImDrawList &self) { return (uint64_t)self.IdxBuffer.Data; })                                                       //
        .def_property_readonly("IdxBufferSize", [](ImDrawList &self) { return sizeof(ImDrawIdx) * self.IdxBuffer.Size; })                                            //
        .def_property_readonly("VtxBufferPtr", [](ImDrawList &self) { return (uint64_t)self.VtxBuffer.Data; })                                                       //
        .def_property_readonly("VtxBufferSize", [](ImDrawList &self) { return sizeof(ImDrawVert) * self.VtxBuffer.Size; })                                           //
        // .def_property_readonly("IdxBuffer", [](ImDrawList &self) { return std::vector<ImDrawIdx>(self.IdxBuffer.Data, self.IdxBuffer.Data + self.IdxBuffer.Size); })  //
        // .def_property_readonly("VtxBuffer", [](ImDrawList &self) { return std::vector<ImDrawVert>(self.VtxBuffer.Data, self.VtxBuffer.Data + self.VtxBuffer.Size); }) //
        ;

    py::class_<ImDrawData>(m, "DrawData")
        .def_property_readonly("CmdLists",
                               [](ImDrawData &self) {
                                   std::vector<ImDrawList *> out;
                                   for (int i = 0; i < self.CmdListsCount; i++) {
                                       out.push_back((self.CmdLists[i]));
                                   }
                                   return out;
                               })                                  //
        .def_readonly("CmdListsCount", &ImDrawData::CmdListsCount) //
        .def_readonly("CmdListsCount", &ImDrawData::CmdListsCount) //
        .def_readonly("TotalIdxCount", &ImDrawData::TotalIdxCount) //
        .def_readonly("TotalVtxCount", &ImDrawData::TotalVtxCount) //
        .def_readonly("DisplayPos", &ImDrawData::DisplayPos)       //
        .def_readonly("DisplaySize", &ImDrawData::DisplaySize)     //
        // .def_property_readonly("FramebufferScale", &ImDrawData::FramebufferScale) //
        ;

    py::class_<ContextWrapper, std::shared_ptr<ContextWrapper>>(m, "Context") //
        .def(py::init<>())
        .def(py::init<ImGuiContext *, uint64_t>(), "ctx"_a, "hwnd"_a)                     //
        .def("GetFontTexturePtr", &ContextWrapper::GetFontTexturePtr)                     //
        .def("GetFontTextureWidth", &ContextWrapper::GetFontTextureWidth)                 //
        .def("GetFontTextureHeight", &ContextWrapper::GetFontTextureHeight)               //
        .def("GetFontTextureBytesPerPixel", &ContextWrapper::GetFontTextureBytesPerPixel) //
        .def("Destroy", &ContextWrapper::Destroy)                                         //
        ;

    enum _ImGuiWindowFlags {
        _ImGuiWindowFlags_None                      = 0,
        _ImGuiWindowFlags_NoTitleBar                = 1 << 0,
        _ImGuiWindowFlags_NoResize                  = 1 << 1,
        _ImGuiWindowFlags_NoMove                    = 1 << 2,
        _ImGuiWindowFlags_NoScrollbar               = 1 << 3,
        _ImGuiWindowFlags_NoScrollWithMouse         = 1 << 4,
        _ImGuiWindowFlags_NoCollapse                = 1 << 5,
        _ImGuiWindowFlags_AlwaysAutoResize          = 1 << 6,
        _ImGuiWindowFlags_NoBackground              = 1 << 7,
        _ImGuiWindowFlags_NoSavedSettings           = 1 << 8,
        _ImGuiWindowFlags_NoMouseInputs             = 1 << 9,
        _ImGuiWindowFlags_MenuBar                   = 1 << 10,
        _ImGuiWindowFlags_HorizontalScrollbar       = 1 << 11,
        _ImGuiWindowFlags_NoFocusOnAppearing        = 1 << 12,
        _ImGuiWindowFlags_NoBringToFrontOnFocus     = 1 << 13,
        _ImGuiWindowFlags_AlwaysVerticalScrollbar   = 1 << 14,
        _ImGuiWindowFlags_AlwaysHorizontalScrollbar = 1 << 15,
        _ImGuiWindowFlags_AlwaysUseWindowPadding    = 1 << 16,
        _ImGuiWindowFlags_NoNavInputs               = 1 << 18,
        _ImGuiWindowFlags_NoNavFocus                = 1 << 19,
        _ImGuiWindowFlags_UnsavedDocument           = 1 << 20,
        _ImGuiWindowFlags_NoNav                     = ImGuiWindowFlags_NoNavInputs | ImGuiWindowFlags_NoNavFocus,
        _ImGuiWindowFlags_NoDecoration              = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse,
        _ImGuiWindowFlags_NoInputs                  = ImGuiWindowFlags_NoMouseInputs | ImGuiWindowFlags_NoNavInputs | ImGuiWindowFlags_NoNavFocus,
        _ImGuiWindowFlags_NavFlattened              = 1 << 23,
        _ImGuiWindowFlags_ChildWindow               = 1 << 24,
        _ImGuiWindowFlags_Tooltip                   = 1 << 25,
        _ImGuiWindowFlags_Popup                     = 1 << 26,
        _ImGuiWindowFlags_Modal                     = 1 << 27,
        _ImGuiWindowFlags_ChildMenu                 = 1 << 28
    };

    enum _ImGuiCond { _ImGuiCond_None = 0, _ImGuiCond_Always = 1 << 0, _ImGuiCond_Once = 1 << 1, _ImGuiCond_FirstUse = 1 << 2, _ImGuiCond_Appearing = 1 << 3 };

    py::enum_<_ImGuiCond>(m, "ImGuiCond", py::arithmetic())
        .value("None", _ImGuiCond_None)
        .value("Always", _ImGuiCond_Always)
        .value("Once", _ImGuiCond_Once)
        .value("FirstUse", _ImGuiCond_FirstUse)
        .value("Appearing", _ImGuiCond_Appearing);

    py::enum_<_ImGuiWindowFlags>(m, "WindowFlags", py::arithmetic())
        .value("None", _ImGuiWindowFlags_None)
        .value("NoTitleBar", _ImGuiWindowFlags_NoTitleBar)
        .value("NoResize", _ImGuiWindowFlags_NoResize)
        .value("NoMove", _ImGuiWindowFlags_NoMove)
        .value("NoScrollbar", _ImGuiWindowFlags_NoScrollbar)
        .value("NoScrollWithMouse", _ImGuiWindowFlags_NoScrollWithMouse)
        .value("NoCollapse", _ImGuiWindowFlags_NoCollapse)
        .value("AlwaysAutoResize", _ImGuiWindowFlags_AlwaysAutoResize)
        .value("NoBackground", _ImGuiWindowFlags_NoBackground)
        .value("NoSavedSettings", _ImGuiWindowFlags_NoSavedSettings)
        .value("NoMouseInputs", _ImGuiWindowFlags_NoMouseInputs)
        .value("MenuBar", _ImGuiWindowFlags_MenuBar)
        .value("HorizontalScrollbar", _ImGuiWindowFlags_HorizontalScrollbar)
        .value("NoFocusOnAppearing", _ImGuiWindowFlags_NoFocusOnAppearing)
        .value("NoBringToFrontOnFocus", _ImGuiWindowFlags_NoBringToFrontOnFocus)
        .value("AlwaysVerticalScrollbar", _ImGuiWindowFlags_AlwaysVerticalScrollbar)
        .value("AlwaysHorizontalScrollbar", _ImGuiWindowFlags_AlwaysHorizontalScrollbar)
        .value("AlwaysUseWindowPadding", _ImGuiWindowFlags_AlwaysUseWindowPadding)
        .value("NoNavInputs", _ImGuiWindowFlags_NoNavInputs)
        .value("NoNavFocus", _ImGuiWindowFlags_NoNavFocus)
        .value("UnsavedDocument", _ImGuiWindowFlags_UnsavedDocument)
        .value("NavFlattened", _ImGuiWindowFlags_NavFlattened)
        .value("ChildWindow", _ImGuiWindowFlags_ChildWindow)
        .value("Tooltip", _ImGuiWindowFlags_Tooltip)
        .value("Popup", _ImGuiWindowFlags_Popup)
        .value("Modal", _ImGuiWindowFlags_Modal)
        .value("ChildMenu", _ImGuiWindowFlags_ChildMenu);

    m.def(
        "GetDrawData", []() { return (ImGui::GetDrawData()); }, py::return_value_policy::reference);
    m.def(
        "Begin", [](const char *name, std::optional<bool> open, int flags) { return ImGui::Begin(name, open ? &open.value() : nullptr, flags); }, "name"_a, "open"_a = std::nullopt,
        "flags"_a = _ImGuiWindowFlags_None);
    m.def("End", &ImGui::End);
    // m.def("BeginChild", &ImGui::BeginChild);
    // m.def("EndChild", &ImGui::EndChild);
    // m.def("BeginGroup", &ImGui::BeginGroup);
    // m.def("EndGroup", &ImGui::EndGroup);
    m.def("SetCursorPos", &ImGui::SetCursorPos);
    m.def("SetCursorScreenPos", &ImGui::SetCursorScreenPos);
    m.def("SetCursorPosX", &ImGui::SetCursorPosX);
    m.def("SetCursorPosY", &ImGui::SetCursorPosY);
    // m.def("GetCurrentContext", []() { return std::make_shared<ContextWrapper>(ImGui::GetCurrentContext()); });
    m.def("SetCurrentContext", [](std::shared_ptr<ContextWrapper> ctx) { ImGui::SetCurrentContext(ctx->ctx); });
    m.def("CreateContext", [](uint64_t hwnd) { return std::make_shared<ContextWrapper>(ImGui::CreateContext(), hwnd); });
    m.def("DestroyContext", [](std::shared_ptr<ContextWrapper> ctx) { ctx->Destroy(); });
    m.def("Render", &ImGui::Render);
    m.def("NewFrame", &ImGui::NewFrame);
    m.def("SetNextWindowSize", &ImGui::SetNextWindowSize, "size"_a, "cond"_a = _ImGuiCond_None);
    // m.def("Initialize", &ImGui::Initialize);
    // m.def("Shutdown", &ImGui::Shutdown);
}

PYBIND11_MODULE(imgui, m) { export_imgui_0(m); }
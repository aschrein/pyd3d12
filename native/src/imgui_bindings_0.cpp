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
        // io.MouseDrawCursor         = true;
        unsigned char *out_pixels = {};
        io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
        io.BackendFlags |= ImGuiBackendFlags_RendererHasViewports;

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
    void SetDisplaySize(int width, int height) {
        ImGui::SetCurrentContext(ctx);
        auto &io       = ImGui::GetIO();
        io.DisplaySize = ImVec2((float)width, (float)height);
    }
    void OnMouseMotion(int x, int y) {
        ImGui::SetCurrentContext(ctx);
        auto &io = ImGui::GetIO();
        io.AddMousePosEvent((float)x, (float)y);
    }
    void OnMousePress(int button) {
        ImGui::SetCurrentContext(ctx);
        auto &io = ImGui::GetIO();
        io.AddMouseButtonEvent(button, true);
    }
    void OnMouseRelease(int button) {
        ImGui::SetCurrentContext(ctx);
        auto &io = ImGui::GetIO();
        io.AddMouseButtonEvent(button, false);
    }
    void OnMouseScroll(float x, float y) {
        ImGui::SetCurrentContext(ctx);
        auto &io = ImGui::GetIO();
        io.AddMouseWheelEvent((float)x, (float)y);
    }
    void OnKeyPress(int key) {
        ImGui::SetCurrentContext(ctx);
        auto &io = ImGui::GetIO();
        io.AddKeyEvent((ImGuiKey)key, true);
    }
    void OnKeyRelease(int key) {
        ImGui::SetCurrentContext(ctx);
        auto &io = ImGui::GetIO();
        io.AddKeyEvent((ImGuiKey)key, false);
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
        .def("OnMouseMotion", &ContextWrapper::OnMouseMotion, "x"_a, "y"_a)               //
        .def("OnMousePress", &ContextWrapper::OnMousePress, "button"_a)                   //
        .def("OnMouseRelease", &ContextWrapper::OnMouseRelease, "button"_a)               //
        .def("OnMouseScroll", &ContextWrapper::OnMouseScroll, "x"_a, "y"_a)               //
        .def("OnKeyPress", &ContextWrapper::OnKeyPress, "key"_a)                          //
        .def("OnKeyRelease", &ContextWrapper::OnKeyRelease, "key"_a)                      //
        .def("SetDisplaySize", &ContextWrapper::SetDisplaySize, "width"_a, "height"_a)    //
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

    enum _ImGuiKey {
        _ImGuiKey_Tab            = ImGuiKey_Tab, // == ImGuiKey_NamedKey_BEGIN
        _ImGuiKey_LeftArrow      = ImGuiKey_LeftArrow,
        _ImGuiKey_RightArrow     = ImGuiKey_RightArrow,
        _ImGuiKey_UpArrow        = ImGuiKey_UpArrow,
        _ImGuiKey_DownArrow      = ImGuiKey_DownArrow,
        _ImGuiKey_PageUp         = ImGuiKey_PageUp,
        _ImGuiKey_PageDown       = ImGuiKey_PageDown,
        _ImGuiKey_Home           = ImGuiKey_Home,
        _ImGuiKey_End            = ImGuiKey_End,
        _ImGuiKey_Insert         = ImGuiKey_Insert,
        _ImGuiKey_Delete         = ImGuiKey_Delete,
        _ImGuiKey_Backspace      = ImGuiKey_Backspace,
        _ImGuiKey_Space          = ImGuiKey_Space,
        _ImGuiKey_Enter          = ImGuiKey_Enter,
        _ImGuiKey_Escape         = ImGuiKey_Escape,
        _ImGuiKey_LeftCtrl       = ImGuiKey_LeftCtrl,
        _ImGuiKey_LeftShift      = ImGuiKey_LeftShift,
        _ImGuiKey_LeftAlt        = ImGuiKey_LeftAlt,
        _ImGuiKey_LeftSuper      = ImGuiKey_LeftSuper,
        _ImGuiKey_RightCtrl      = ImGuiKey_RightCtrl,
        _ImGuiKey_RightShift     = ImGuiKey_RightShift,
        _ImGuiKey_RightAlt       = ImGuiKey_RightAlt,
        _ImGuiKey_RightSuper     = ImGuiKey_RightSuper,
        _ImGuiKey_Menu           = ImGuiKey_Menu,
        _ImGuiKey_0              = ImGuiKey_0,
        _ImGuiKey_1              = ImGuiKey_1,
        _ImGuiKey_2              = ImGuiKey_2,
        _ImGuiKey_3              = ImGuiKey_3,
        _ImGuiKey_4              = ImGuiKey_4,
        _ImGuiKey_5              = ImGuiKey_5,
        _ImGuiKey_6              = ImGuiKey_6,
        _ImGuiKey_7              = ImGuiKey_7,
        _ImGuiKey_8              = ImGuiKey_8,
        _ImGuiKey_9              = ImGuiKey_9,
        _ImGuiKey_A              = ImGuiKey_A,
        _ImGuiKey_B              = ImGuiKey_B,
        _ImGuiKey_C              = ImGuiKey_C,
        _ImGuiKey_D              = ImGuiKey_D,
        _ImGuiKey_E              = ImGuiKey_E,
        _ImGuiKey_F              = ImGuiKey_F,
        _ImGuiKey_G              = ImGuiKey_G,
        _ImGuiKey_H              = ImGuiKey_H,
        _ImGuiKey_I              = ImGuiKey_I,
        _ImGuiKey_J              = ImGuiKey_J,
        _ImGuiKey_K              = ImGuiKey_K,
        _ImGuiKey_L              = ImGuiKey_L,
        _ImGuiKey_M              = ImGuiKey_M,
        _ImGuiKey_N              = ImGuiKey_N,
        _ImGuiKey_O              = ImGuiKey_O,
        _ImGuiKey_P              = ImGuiKey_P,
        _ImGuiKey_Q              = ImGuiKey_Q,
        _ImGuiKey_R              = ImGuiKey_R,
        _ImGuiKey_S              = ImGuiKey_S,
        _ImGuiKey_T              = ImGuiKey_T,
        _ImGuiKey_U              = ImGuiKey_U,
        _ImGuiKey_V              = ImGuiKey_V,
        _ImGuiKey_W              = ImGuiKey_W,
        _ImGuiKey_X              = ImGuiKey_X,
        _ImGuiKey_Y              = ImGuiKey_Y,
        _ImGuiKey_Z              = ImGuiKey_Z,
        _ImGuiKey_F1             = ImGuiKey_F1,
        _ImGuiKey_F2             = ImGuiKey_F2,
        _ImGuiKey_F3             = ImGuiKey_F3,
        _ImGuiKey_F4             = ImGuiKey_F4,
        _ImGuiKey_F5             = ImGuiKey_F5,
        _ImGuiKey_F6             = ImGuiKey_F6,
        _ImGuiKey_F7             = ImGuiKey_F7,
        _ImGuiKey_F8             = ImGuiKey_F8,
        _ImGuiKey_F9             = ImGuiKey_F9,
        _ImGuiKey_F10            = ImGuiKey_F10,
        _ImGuiKey_F11            = ImGuiKey_F11,
        _ImGuiKey_F12            = ImGuiKey_F12,
        _ImGuiKey_F13            = ImGuiKey_F13,
        _ImGuiKey_F14            = ImGuiKey_F14,
        _ImGuiKey_F15            = ImGuiKey_F15,
        _ImGuiKey_F16            = ImGuiKey_F16,
        _ImGuiKey_F17            = ImGuiKey_F17,
        _ImGuiKey_F18            = ImGuiKey_F18,
        _ImGuiKey_F19            = ImGuiKey_F19,
        _ImGuiKey_F20            = ImGuiKey_F20,
        _ImGuiKey_F21            = ImGuiKey_F21,
        _ImGuiKey_F22            = ImGuiKey_F22,
        _ImGuiKey_F23            = ImGuiKey_F23,
        _ImGuiKey_F24            = ImGuiKey_F24,
        _ImGuiKey_Apostrophe     = ImGuiKey_Apostrophe,   // '
        _ImGuiKey_Comma          = ImGuiKey_Comma,        // ,
        _ImGuiKey_Minus          = ImGuiKey_Minus,        // -
        _ImGuiKey_Period         = ImGuiKey_Period,       // .
        _ImGuiKey_Slash          = ImGuiKey_Slash,        // /
        _ImGuiKey_Semicolon      = ImGuiKey_Semicolon,    // ;
        _ImGuiKey_Equal          = ImGuiKey_Equal,        // =
        _ImGuiKey_LeftBracket    = ImGuiKey_LeftBracket,  // [
        _ImGuiKey_Backslash      = ImGuiKey_Backslash,    // \ (this text inhibit multiline comment caused by backslash)
        _ImGuiKey_RightBracket   = ImGuiKey_RightBracket, // ]
        _ImGuiKey_GraveAccent    = ImGuiKey_GraveAccent,  // `
        _ImGuiKey_CapsLock       = ImGuiKey_CapsLock,
        _ImGuiKey_ScrollLock     = ImGuiKey_ScrollLock,
        _ImGuiKey_NumLock        = ImGuiKey_NumLock,
        _ImGuiKey_PrintScreen    = ImGuiKey_PrintScreen,
        _ImGuiKey_Pause          = ImGuiKey_Pause,
        _ImGuiKey_Keypad0        = ImGuiKey_Keypad0,
        _ImGuiKey_Keypad1        = ImGuiKey_Keypad1,
        _ImGuiKey_Keypad2        = ImGuiKey_Keypad2,
        _ImGuiKey_Keypad3        = ImGuiKey_Keypad3,
        _ImGuiKey_Keypad4        = ImGuiKey_Keypad4,
        _ImGuiKey_Keypad5        = ImGuiKey_Keypad5,
        _ImGuiKey_Keypad6        = ImGuiKey_Keypad6,
        _ImGuiKey_Keypad7        = ImGuiKey_Keypad7,
        _ImGuiKey_Keypad8        = ImGuiKey_Keypad8,
        _ImGuiKey_Keypad9        = ImGuiKey_Keypad9,
        _ImGuiKey_KeypadDecimal  = ImGuiKey_KeypadDecimal,
        _ImGuiKey_KeypadDivide   = ImGuiKey_KeypadDivide,
        _ImGuiKey_KeypadMultiply = ImGuiKey_KeypadMultiply,
        _ImGuiKey_KeypadSubtract = ImGuiKey_KeypadSubtract,
        _ImGuiKey_KeypadAdd      = ImGuiKey_KeypadAdd,
        _ImGuiKey_KeypadEnter    = ImGuiKey_KeypadEnter,
        _ImGuiKey_KeypadEqual    = ImGuiKey_KeypadEqual,
        _ImGuiKey_AppBack        = ImGuiKey_AppBack, // Available on some keyboard/mouses. Often referred as "Browser Back"
        _ImGuiKey_AppForward     = ImGuiKey_AppForward,
    };

    py::enum_<_ImGuiKey>(m, "Key", py::arithmetic())
        .value("Tab", _ImGuiKey_Tab)
        .value("LeftArrow", _ImGuiKey_LeftArrow)
        .value("RightArrow", _ImGuiKey_RightArrow)
        .value("UpArrow", _ImGuiKey_UpArrow)
        .value("DownArrow", _ImGuiKey_DownArrow)
        .value("PageUp", _ImGuiKey_PageUp)
        .value("PageDown", _ImGuiKey_PageDown)
        .value("Home", _ImGuiKey_Home)
        .value("End", _ImGuiKey_End)
        .value("Insert", _ImGuiKey_Insert)
        .value("Delete", _ImGuiKey_Delete)
        .value("Backspace", _ImGuiKey_Backspace)
        .value("Space", _ImGuiKey_Space)
        .value("Enter", _ImGuiKey_Enter)
        .value("Escape", _ImGuiKey_Escape)
        .value("LeftCtrl", _ImGuiKey_LeftCtrl)
        .value("LeftShift", _ImGuiKey_LeftShift)
        .value("LeftAlt", _ImGuiKey_LeftAlt)
        .value("LeftSuper", _ImGuiKey_LeftSuper)
        .value("RightCtrl", _ImGuiKey_RightCtrl)
        .value("RightShift", _ImGuiKey_RightShift)
        .value("RightAlt", _ImGuiKey_RightAlt)
        .value("RightSuper", _ImGuiKey_RightSuper)
        .value("Menu", _ImGuiKey_Menu)
        .value("_0", _ImGuiKey_0)
        .value("_1", _ImGuiKey_1)
        .value("_2", _ImGuiKey_2)
        .value("_3", _ImGuiKey_3)
        .value("_4", _ImGuiKey_4)
        .value("_5", _ImGuiKey_5)
        .value("_6", _ImGuiKey_6)
        .value("_7", _ImGuiKey_7)
        .value("_8", _ImGuiKey_8)
        .value("_9", _ImGuiKey_9)
        .value("A", _ImGuiKey_A)
        .value("B", _ImGuiKey_B)
        .value("C", _ImGuiKey_C)
        .value("D", _ImGuiKey_D)
        .value("E", _ImGuiKey_E)
        .value("F", _ImGuiKey_F)
        .value("G", _ImGuiKey_G)
        .value("H", _ImGuiKey_H)
        .value("I", _ImGuiKey_I)
        .value("J", _ImGuiKey_J)
        .value("K", _ImGuiKey_K)
        .value("L", _ImGuiKey_L)
        .value("M", _ImGuiKey_M)
        .value("N", _ImGuiKey_N)
        .value("O", _ImGuiKey_O)
        .value("P", _ImGuiKey_P)
        .value("Q", _ImGuiKey_Q)
        .value("R", _ImGuiKey_R)
        .value("S", _ImGuiKey_S)
        .value("T", _ImGuiKey_T)
        .value("U", _ImGuiKey_U)
        .value("V", _ImGuiKey_V)
        .value("W", _ImGuiKey_W)
        .value("X", _ImGuiKey_X)
        .value("Y", _ImGuiKey_Y)
        .value("Z", _ImGuiKey_Z)
        .value("F1", _ImGuiKey_F1)
        .value("F2", _ImGuiKey_F2)
        .value("F3", _ImGuiKey_F3)
        .value("F4", _ImGuiKey_F4)
        .value("F5", _ImGuiKey_F5)
        .value("F6", _ImGuiKey_F6)
        .value("F7", _ImGuiKey_F7)
        .value("F8", _ImGuiKey_F8)
        .value("F9", _ImGuiKey_F9)
        .value("F10", _ImGuiKey_F10)
        .value("F11", _ImGuiKey_F11)
        .value("F12", _ImGuiKey_F12)
        .value("F13", _ImGuiKey_F13)
        .value("F14", _ImGuiKey_F14)
        .value("F15", _ImGuiKey_F15)
        .value("F16", _ImGuiKey_F16)
        .value("F17", _ImGuiKey_F17)
        .value("F18", _ImGuiKey_F18)
        .value("F19", _ImGuiKey_F19)
        .value("F20", _ImGuiKey_F20)
        .value("F21", _ImGuiKey_F21)
        .value("F22", _ImGuiKey_F22)
        .value("F23", _ImGuiKey_F23)
        .value("F24", _ImGuiKey_F24)
        .value("Apostrophe", _ImGuiKey_Apostrophe)
        .value("Comma", _ImGuiKey_Comma)
        .value("Minus", _ImGuiKey_Minus)
        .value("Period", _ImGuiKey_Period)
        .value("Slash", _ImGuiKey_Slash)
        .value("Semicolon", _ImGuiKey_Semicolon)
        .value("Equal", _ImGuiKey_Equal)
        .value("LeftBracket", _ImGuiKey_LeftBracket)
        .value("Backslash", _ImGuiKey_Backslash)
        .value("RightBracket", _ImGuiKey_RightBracket)
        .value("GraveAccent", _ImGuiKey_GraveAccent)
        .value("CapsLock", _ImGuiKey_CapsLock)
        .value("ScrollLock", _ImGuiKey_ScrollLock)
        .value("NumLock", _ImGuiKey_NumLock)
        .value("PrintScreen", _ImGuiKey_PrintScreen)
        .value("Pause", _ImGuiKey_Pause)
        .value("Keypad0", _ImGuiKey_Keypad0)
        .value("Keypad1", _ImGuiKey_Keypad1)
        .value("Keypad2", _ImGuiKey_Keypad2)
        .value("Keypad3", _ImGuiKey_Keypad3)
        .value("Keypad4", _ImGuiKey_Keypad4)
        .value("Keypad5", _ImGuiKey_Keypad5)
        .value("Keypad6", _ImGuiKey_Keypad6)
        .value("Keypad7", _ImGuiKey_Keypad7)
        .value("Keypad8", _ImGuiKey_Keypad8)
        .value("Keypad9", _ImGuiKey_Keypad9)
        .value("KeypadDecimal", _ImGuiKey_KeypadDecimal)
        .value("KeypadDivide", _ImGuiKey_KeypadDivide)
        .value("KeypadMultiply", _ImGuiKey_KeypadMultiply)
        .value("KeypadSubtract", _ImGuiKey_KeypadSubtract)
        .value("KeypadAdd", _ImGuiKey_KeypadAdd)
        .value("KeypadEnter", _ImGuiKey_KeypadEnter)
        .value("KeypadEqual", _ImGuiKey_KeypadEqual)
        .value("AppBack", _ImGuiKey_AppBack)
        .value("AppForward", _ImGuiKey_AppForward);

    enum _ImGuiCond { _ImGuiCond_None = 0, _ImGuiCond_Always = 1 << 0, _ImGuiCond_Once = 1 << 1, _ImGuiCond_FirstUse = 1 << 2, _ImGuiCond_Appearing = 1 << 3 };

    py::enum_<_ImGuiCond>(m, "Cond", py::arithmetic())
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
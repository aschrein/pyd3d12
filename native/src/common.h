/**
*  MIT License
*
*  Copyright (c) 2025 Anton Schreiner
*
*  Permission is hereby granted, free of charge, to any person obtaining a copy
*  of this software and associated documentation files (the "Software"), to deal
*  in the Software without restriction, including without limitation the rights
*  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*  copies of the Software, and to permit persons to whom the Software is
*  furnished to do so, subject to the following conditions:
*
*  The above copyright notice and this permission notice shall be included in all
*  copies or substantial portions of the Software.
*
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*  SOFTWARE.
*/

#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)

#define ASSERT_HRESULT_PANIC(hr)                                                                                                                                                   \
    if (FAILED(hr)) {                                                                                                                                                              \
        std::stringstream ss;                                                                                                                                                      \
        ss << std::hex << hr;                                                                                                                                                      \
        throw std::runtime_error("" __FILE__ ":" STRINGIFY(__LINE__) " HRESULT: 0x" + ss.str());                                                                                   \
    }

#define ASSERT_PANIC(cond)                                                                                                                                                         \
    if (!(cond)) {                                                                                                                                                                 \
        throw std::runtime_error("" __FILE__ ":" STRINGIFY(__LINE__) " " #cond);                                                                                                   \
    }

void export_d3d12_0(py::module &m);
void export_imgui_0(py::module &m);

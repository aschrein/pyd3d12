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

#include <Windows.h>
#define USE_PIX
#include <pix/pix3.h>

PYBIND11_MODULE(pix, m) {

    m.def("PIXLoadLatestWinPixGpuCapturerLibrary", []() {
        HMODULE module = PIXLoadLatestWinPixGpuCapturerLibrary();
        ASSERT_PANIC(module != nullptr && "Failed to load WinPixGpuCapturer.dll");
    });
    m.def("PIXBeginCapture", []() { PIXBeginCapture(PIX_CAPTURE_GPU, nullptr); });
    m.def("PIXEndCapture", []() { PIXEndCapture(false); });
}
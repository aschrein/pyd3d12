// MIT License
//
// Copyright (c) 2025 Anton Schreiner
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "utils.hlsli"

struct WeightsAndBiasesDescriptor {
    ByteAddressBuffer initializer;
    i32               weights_offset_bytes;
    i32               biases_offset_bytes;
};

template <int DstChannels, int SrcChannels> static void conv_1x1_fp16(inout f32 dst[DstChannels], inout f16 src[SrcChannels], WeightsAndBiasesDescriptor wnb) {
    _Static_assert(DstChannels % 16 == 0, "DstChannels must be a multiple of 16");
    _Static_assert(SrcChannels % 16 == 0, "SrcChannels must be a multiple of 16");

    if (wnb.biases_offset_bytes >= i32(0)) {
        ifor(DstChannels / 8) {
            const u32x4 pack   = wnb.initializer.Load<u32x4>(wnb.biases_offset_bytes + i * 16);
            const f16x8 biases = unpack_u32x4_to_f16x8(pack);
            jfor(8) { dst[i * 8 + j] = f32(biases.m[j]); }
        }
    } else {
        jfor(DstChannels) { dst[j] = f32(0.0); }
    }

    ifor(DstChannels) {
        jfor(SrcChannels / 8) {
            const u32x4 pack    = wnb.initializer.Load<u32x4>(wnb.weights_offset_bytes + i * 32 + j * 4);
            const f16x8 weights = unpack_u32x4_to_f16x8(pack);
            kfor(4) { dst[i * 8 + j] = dot2add(f16x2(src[j * 8 + 2 * k + 0], src[j * 8 + 2 * k + 1]), f16x2(weights.m[2 * k], weights.m[2 * k + 1]), dst[i * 8 + j]); }
        }
    }
}
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

#ifndef UTILS_HLSLI
#define UTILS_HLSLI

using f32     = float;
using f32x2   = float2;
using f32x3   = float3;
using f32x4   = float4;
using f16     = half;
using f16x2   = half2;
using f16x3   = half3;
using f16x4   = half4;
using i32     = int;
using i32x2   = int2;
using i32x3   = int3;
using i32x4   = int4;
using u32     = uint;
using u32x2   = uint2;
using u32x3   = uint3;
using u32x4   = uint4;
using f32x2x2 = float2x2;
using f32x2x3 = float2x3;
using f32x2x4 = float2x4;
using f32x3x2 = float3x2;
using f32x3x3 = float3x3;
using f32x3x4 = float3x4;
using f32x4x2 = float4x2;
using f32x4x3 = float4x3;
using f32x4x4 = float4x4;
using f16x2x2 = half2x2;
using f16x2x3 = half2x3;
using f16x2x4 = half2x4;
using f16x3x2 = half3x2;
using f16x3x3 = half3x3;
using f16x3x4 = half3x4;
using f16x4x2 = half4x2;
using f16x4x3 = half4x3;
using f16x4x4 = half4x4;
using i32x2x2 = int2x2;
using i16     = int16_t;
using i16x2   = int16_t2;
using i16x3   = int16_t3;
using i16x4   = int16_t4;
using u16     = uint16_t;
using u16x2   = uint16_t2;
using u16x3   = uint16_t3;
using u16x4   = uint16_t4;

#define asf32(x) asfloat(x)
#define asu32(x) asuint(x)
#define asu16(x) asuint16(x)

#define ifor(N) for (int i = 0; i < (int)(N); i++)
#define ufor(N) for (uint i = 0; i < N; i++)
#define ffor(N) for (float i = 0; i < N; i++)
#define jfor(N) for (int j = 0; j < (int)(N); j++)
#define kfor(N) for (int k = 0; k < (int)(N); k++)
#define xfor(N) for (int x = 0; x < (int)(N); x++)
#define yfor(N) for (int y = 0; y < (int)(N); y++)
#define zfor(N) for (int z = 0; z < (int)(N); z++)

static u32 pack_f16x2_to_u32(f16x2 v) {
    u32 result;
    result = f32tof16(v.x) | (f32tof16(v.y) << 16);
    return result;
}
static f16x2 unpack_u32_to_f16x2(u32 v) {
    f16x2 result;
    result.x = (f16)f16tof32(v & 0xFFFF);
    result.y = (f16)f16tof32((v >> 16) & 0xFFFF);
    return result;
}
struct f16x8 {
    f16 m[8];
};
static f16x8 unpack_u32x4_to_f16x8(u32x4 v) {
    f16x8 result;
    result.m[0] = (f16)f16tof32(v.x & 0xFFFF);
    result.m[1] = (f16)f16tof32((v.x >> 16) & 0xFFFF);
    result.m[2] = (f16)f16tof32(v.y & 0xFFFF);
    result.m[3] = (f16)f16tof32((v.y >> 16) & 0xFFFF);
    result.m[4] = (f16)f16tof32(v.z & 0xFFFF);
    result.m[5] = (f16)f16tof32((v.z >> 16) & 0xFFFF);
    result.m[6] = (f16)f16tof32(v.w & 0xFFFF);
    result.m[7] = (f16)f16tof32((v.w >> 16) & 0xFFFF);
    return result;
}

static u32 fp8_m4e3fn_to_fp16(u32 fp8) {
    // Extract components from FP8
    u32 sign     = (fp8 >> 7) & 1;   // Sign bit
    u32 exponent = (fp8 >> 3) & 0xF; // 4-bit exponent
    u32 mantissa = fp8 & 0x7;        // 3-bit mantissa

    if (exponent == 0) // Zero or subnormal
    {
        if (mantissa == 0) {
            // Zero: only sign bit set
            return (sign << 15);
        } else {
            // Subnormal: value = (-1)^sign * 2^(-6) * (mantissa / 8)
            // Convert to 32-bit float for precise conversion, then to FP16
            float value = (sign ? -1.0f : 1.0f) * pow(2.0f, -6.0f) * (float(mantissa) / 8.0f);
            return f32tof16(value);
        }
    } else if (exponent == 15) // Infinity or NaN
    {
        if (mantissa == 0) {
            // Infinity: exponent = 31, mantissa = 0
            return (sign << 15) | (31 << 10);
        } else {
            // NaN: exponent = 31, mantissa non-zero (copy FP8 mantissa)
            return (sign << 15) | (31 << 10) | (mantissa << 7);
        }
    } else // Normal number
    {
        // Adjust exponent: FP16_bias (15) - FP8_bias (7) = 8
        int expAdjusted = int(exponent) + 8;
        if (expAdjusted > 30) {
            // Overflow to infinity
            return (sign << 15) | (31 << 10);
        } else {
            // Normal FP16: shift mantissa to align with 10-bit field
            u32 exponentFP16 = u32(expAdjusted);
            u32 mantissaFP16 = mantissa << 7; // 3 bits to 10 bits, pad with zeros
            return (sign << 15) | (exponentFP16 << 10) | mantissaFP16;
        }
    }
}

static u32 fp16_to_fp8_m4e3fn(u32 fp16) {
    // Convert FP16 to 32-bit float for special case handling
    f16 value = (f16)f16tof32(fp16);
    u32 sign  = (fp16 >> 15) & 1;

    if (isinf(value)) {
        // Infinity: exponent = 15, mantissa = 0
        return 0x7F; // All 1s
    } else if (isnan(value)) {
        // NaN: exponent = 15, mantissa non-zero
        return 0x7F; // All 1s
    } else if (value == 0.0f) {
        // Zero: only sign bit set
        return (sign << 7);
    } else {
        // Extract components from FP16
        u32 exponentFP16 = (fp16 >> 10) & 0x1F;
        u32 mantissaFP16 = fp16 & 0x3FF;

        if (exponentFP16 == 0) // FP16 subnormal
        {
            // Too small for FP8’s range, clamp to zero
            return (sign << 7);
        } else // Normal FP16 number
        {
            // True exponent in FP16: exponent - 15
            int trueExp = int(exponentFP16) - 15;
            if (trueExp < -7) {
                // Underflow to zero
                return (sign << 7);
            } else if (trueExp > 7) {
                // Overflow to infinity
                return (sign << 7) | (15 << 3);
            } else {
                // Normal FP8: adjust exponent and truncate mantissa
                u32 exponentFP8 = u32(trueExp + 7);          // Bias 7 for FP8
                u32 mantissaFP8 = (mantissaFP16 >> 7) & 0x7; // Top 3 bits of 10
                return (sign << 7) | (exponentFP8 << 3) | mantissaFP8;
            }
        }
    }
}

static uint xxhash(in uint p) {
    const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;

    uint h32 = p + PRIME32_5;
    h32      = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32      = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32      = PRIME32_3 * (h32 ^ (h32 >> 13));

    return h32 ^ (h32 >> 16);
}

static float3 random_color(uint a) {
    a = xxhash(a);
    return float3((a & 0xff) / 255.0f, ((a >> 8) & 0xff) / 255.0f, ((a >> 16) & 0xff) / 255.0f);
}

static uint64_t u32x2_to_u64(uint2 v) { return (uint64_t(v.y) << uint64_t(32)) | uint64_t(v.x); }

// https://www.shadertoy.com/view/WslGRN
static float3 heatmap(float t) {
    t            = saturate(t);
    float  level = t * 3.14159265 / 2.;
    float3 col;
    col.r = sin(level);
    col.g = sin(level * 2.);
    col.b = cos(level);
    return col;
}

static float3x3 GetTBN(float3 normal) {
    float3 up        = abs(normal.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tangent   = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);
    return float3x3(tangent, bitangent, normal);
}

static float2 SampleRandomCircle(float2 xi) {
    float r     = sqrt(xi.x);
    float theta = 2 * 3.14159265359 * xi.y;
    return float2(r * cos(theta), r * sin(theta));
}

#endif // !UTILS_HLSLI

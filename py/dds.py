# MIT License
# 
# Copyright (c) 2025 Anton Schreiner
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This file describes dds header format and some utilities to manipulate data in dds files, save, load, etc.
"""

from enum import Enum, IntEnum
from struct import *
import ctypes
import struct
import numpy as np

# https://learn.microsoft.com/en-us/windows/win32/api/dxgiformat/ne-dxgiformat-dxgi_format
class DXGI_FORMAT(IntEnum):
    UNKNOWN = 0,
    R32G32B32A32_TYPELESS = 1,
    R32G32B32A32_FLOAT = 2,
    R32G32B32A32_UINT = 3,
    R32G32B32A32_SINT = 4,
    R32G32B32_TYPELESS = 5,
    R32G32B32_FLOAT = 6,
    R32G32B32_UINT = 7,
    R32G32B32_SINT = 8,
    R16G16B16A16_TYPELESS = 9,
    R16G16B16A16_FLOAT = 10,
    R16G16B16A16_UNORM = 11,
    R16G16B16A16_UINT = 12,
    R16G16B16A16_SNORM = 13,
    R16G16B16A16_SINT = 14,
    R32G32_TYPELESS = 15,
    R32G32_FLOAT = 16,
    R32G32_UINT = 17,
    R32G32_SINT = 18,
    R32G8X24_TYPELESS = 19,
    D32_FLOAT_S8X24_UINT = 20,
    R32_FLOAT_X8X24_TYPELESS = 21,
    X32_TYPELESS_G8X24_UINT = 22,
    R10G10B10A2_TYPELESS = 23,
    R10G10B10A2_UNORM = 24,
    R10G10B10A2_UINT = 25,
    R11G11B10_FLOAT = 26,
    R8G8B8A8_TYPELESS = 27,
    R8G8B8A8_UNORM = 28,
    R8G8B8A8_UNORM_SRGB = 29,
    R8G8B8A8_UINT = 30,
    R8G8B8A8_SNORM = 31,
    R8G8B8A8_SINT = 32,
    R16G16_TYPELESS = 33,
    R16G16_FLOAT = 34,
    R16G16_UNORM = 35,
    R16G16_UINT = 36,
    R16G16_SNORM = 37,
    R16G16_SINT = 38,
    R32_TYPELESS = 39,
    D32_FLOAT = 40,
    R32_FLOAT = 41,
    R32_UINT = 42,
    R32_SINT = 43,
    R24G8_TYPELESS = 44,
    D24_UNORM_S8_UINT = 45,
    R24_UNORM_X8_TYPELESS = 46,
    X24_TYPELESS_G8_UINT = 47,
    R8G8_TYPELESS = 48,
    R8G8_UNORM = 49,
    R8G8_UINT = 50,
    R8G8_SNORM = 51,
    R8G8_SINT = 52,
    R16_TYPELESS = 53,
    R16_FLOAT = 54,
    D16_UNORM = 55,
    R16_UNORM = 56,
    R16_UINT = 57,
    R16_SNORM = 58,
    R16_SINT = 59,
    R8_TYPELESS = 60,
    R8_UNORM = 61,
    R8_UINT = 62,
    R8_SNORM = 63,
    R8_SINT = 64,
    A8_UNORM = 65,
    R1_UNORM = 66,
    R9G9B9E5_SHAREDEXP = 67,
    R8G8_B8G8_UNORM = 68,
    G8R8_G8B8_UNORM = 69,
    BC1_TYPELESS = 70,
    BC1_UNORM = 71,
    BC1_UNORM_SRGB = 72,
    BC2_TYPELESS = 73,
    BC2_UNORM = 74,
    BC2_UNORM_SRGB = 75,
    BC3_TYPELESS = 76,
    BC3_UNORM = 77,
    BC3_UNORM_SRGB = 78,
    BC4_TYPELESS = 79,
    BC4_UNORM = 80,
    BC4_SNORM = 81,
    BC5_TYPELESS = 82,
    BC5_UNORM = 83,
    BC5_SNORM = 84,
    B5G6R5_UNORM = 85,
    B5G5R5A1_UNORM = 86,
    B8G8R8A8_UNORM = 87,
    B8G8R8X8_UNORM = 88,
    R10G10B10_XR_BIAS_A2_UNORM = 89,
    B8G8R8A8_TYPELESS = 90,
    B8G8R8A8_UNORM_SRGB = 91,
    B8G8R8X8_TYPELESS = 92,
    B8G8R8X8_UNORM_SRGB = 93,
    BC6H_TYPELESS = 94,
    BC6H_UF16 = 95,
    BC6H_SF16 = 96,
    BC7_TYPELESS = 97,
    BC7_UNORM = 98,
    BC7_UNORM_SRGB = 99,
    AYUV = 100,
    Y410 = 101,
    Y416 = 102,
    NV12 = 103,
    P010 = 104,
    P016 = 105,
    _420_OPAQUE = 106,
    YUY2 = 107,
    Y210 = 108,
    Y216 = 109,
    NV11 = 110,
    AI44 = 111,
    IA44 = 112,
    P8 = 113,
    A8P8 = 114,
    B4G4R4A4_UNORM = 115,
    P208 = 130,
    V208 = 131,
    V408 = 132,
    FORCE_UINT = 0xffffffff

def dds_is_format_compressed(dxgi_format: DXGI_FORMAT) -> bool:
    return dxgi_format in [
        DXGI_FORMAT.BC1_TYPELESS,
        DXGI_FORMAT.BC1_UNORM,
        DXGI_FORMAT.BC1_UNORM_SRGB,
        DXGI_FORMAT.BC2_TYPELESS,
        DXGI_FORMAT.BC2_UNORM,
        DXGI_FORMAT.BC2_UNORM_SRGB,
        DXGI_FORMAT.BC3_TYPELESS,
        DXGI_FORMAT.BC3_UNORM,
        DXGI_FORMAT.BC3_UNORM_SRGB,
        DXGI_FORMAT.BC4_TYPELESS,
        DXGI_FORMAT.BC4_UNORM,
        DXGI_FORMAT.BC4_SNORM,
        DXGI_FORMAT.BC5_TYPELESS,
        DXGI_FORMAT.BC5_UNORM,
        DXGI_FORMAT.BC5_SNORM,
        DXGI_FORMAT.BC6H_TYPELESS,
        DXGI_FORMAT.BC6H_UF16,
        DXGI_FORMAT.BC6H_SF16,
        DXGI_FORMAT.BC7_TYPELESS,
        DXGI_FORMAT.BC7_UNORM,
        DXGI_FORMAT.BC7_UNORM_SRGB,
    ]

def dds_get_bytes_per_block(dxgi_format: DXGI_FORMAT) -> int:
    map = {
        DXGI_FORMAT.BC1_TYPELESS: 8,
        DXGI_FORMAT.BC1_UNORM: 8,
        DXGI_FORMAT.BC1_UNORM_SRGB: 8,
        DXGI_FORMAT.BC2_TYPELESS: 16,
        DXGI_FORMAT.BC2_UNORM: 16,
        DXGI_FORMAT.BC2_UNORM_SRGB: 16,
        DXGI_FORMAT.BC3_TYPELESS: 16,
        DXGI_FORMAT.BC3_UNORM: 16,
        DXGI_FORMAT.BC3_UNORM_SRGB: 16,
        DXGI_FORMAT.BC4_TYPELESS: 8,
        DXGI_FORMAT.BC4_UNORM: 8,
        DXGI_FORMAT.BC4_SNORM: 8,
        DXGI_FORMAT.BC5_TYPELESS: 16,
        DXGI_FORMAT.BC5_UNORM: 16,
        DXGI_FORMAT.BC5_SNORM: 16,
        DXGI_FORMAT.BC6H_TYPELESS: 16,
        DXGI_FORMAT.BC6H_UF16: 16,
        DXGI_FORMAT.BC6H_SF16: 16,
        DXGI_FORMAT.BC7_TYPELESS: 16,
        DXGI_FORMAT.BC7_UNORM: 16,
        DXGI_FORMAT.BC7_UNORM_SRGB: 16
        }
    return map[dxgi_format]

def dds_get_bytes_per_pixel(dxgi_format: DXGI_FORMAT) -> int:
    if dds_is_format_compressed(dxgi_format): return dds_get_bytes_per_block(dxgi_format)
    map = {
        DXGI_FORMAT.R32G32B32A32_FLOAT: 16,
        DXGI_FORMAT.R32G32B32A32_UINT: 16,
        DXGI_FORMAT.R32G32B32A32_SINT: 16,
        DXGI_FORMAT.R32G32B32_FLOAT: 12,
        DXGI_FORMAT.R32G32B32_UINT: 12,
        DXGI_FORMAT.R32G32B32_SINT: 12,
        DXGI_FORMAT.R16G16B16A16_FLOAT: 8,
        DXGI_FORMAT.R16G16B16A16_UNORM: 8,
        DXGI_FORMAT.R16G16B16A16_UINT: 8,
        DXGI_FORMAT.R16G16B16A16_SNORM: 8,
        DXGI_FORMAT.R16G16B16A16_SINT: 8,
        DXGI_FORMAT.R32G32_FLOAT: 8,
        DXGI_FORMAT.R32G32_UINT: 8,
        DXGI_FORMAT.R32G32_SINT: 8,
        DXGI_FORMAT.R10G10B10A2_UNORM: 4,
        DXGI_FORMAT.R10G10B10A2_UINT: 4,
        DXGI_FORMAT.R11G11B10_FLOAT: 4,
        DXGI_FORMAT.R8G8B8A8_UNORM: 4,
        DXGI_FORMAT.R8G8B8A8_UNORM_SRGB: 4,
        DXGI_FORMAT.R8G8B8A8_UINT: 4,
        DXGI_FORMAT.R8G8B8A8_SNORM: 4,
        DXGI_FORMAT.R8G8B8A8_SINT: 4,
        DXGI_FORMAT.R16G16_FLOAT: 4,
        DXGI_FORMAT.R16G16_UNORM: 4,
        DXGI_FORMAT.R16G16_UINT: 4,
        DXGI_FORMAT.R16G16_SNORM: 4,
        DXGI_FORMAT.R16G16_SINT: 4,
        DXGI_FORMAT.R32_FLOAT: 4,
        DXGI_FORMAT.R32_UINT: 4,
        DXGI_FORMAT.R32_SINT: 4,
        DXGI_FORMAT.R8G8_UNORM: 2,
        DXGI_FORMAT.R8G8_UINT: 2,
        DXGI_FORMAT.R8G8_SNORM: 2,
        DXGI_FORMAT.R8G8_SINT: 2,
        DXGI_FORMAT.R16_FLOAT: 2,
        DXGI_FORMAT.R16_UNORM: 2,
        DXGI_FORMAT.R16_UINT: 2,
        DXGI_FORMAT.R16_SNORM: 2,
        DXGI_FORMAT.R16_SINT: 2,
        DXGI_FORMAT.R8_UNORM: 1,
        DXGI_FORMAT.R8_UINT: 1,
        DXGI_FORMAT.R8_SNORM: 1,
        DXGI_FORMAT.R8_SINT: 1,
        DXGI_FORMAT.A8_UNORM: 1,
        DXGI_FORMAT.R1_UNORM: 1,
        DXGI_FORMAT.R9G9B9E5_SHAREDEXP: 4,
        DXGI_FORMAT.R8G8_B8G8_UNORM: 4,
        DXGI_FORMAT.G8R8_G8B8_UNORM: 4,
        DXGI_FORMAT.B5G6R5_UNORM: 2,
        DXGI_FORMAT.B5G5R5A1_UNORM: 2,
        DXGI_FORMAT.B8G8R8A8_UNORM: 4,
        DXGI_FORMAT.B8G8R8X8_UNORM: 4,
        DXGI_FORMAT.R10G10B10_XR_BIAS_A2_UNORM: 4,
        DXGI_FORMAT.B8G8R8A8_TYPELESS: 4,
        DXGI_FORMAT.B8G8R8A8_UNORM_SRGB: 4,
        DXGI_FORMAT.B8G8R8X8_TYPELESS: 4,
        DXGI_FORMAT.B8G8R8X8_UNORM_SRGB: 4,
        DXGI_FORMAT.AYUV: 4,
        DXGI_FORMAT.Y410: 4,
        DXGI_FORMAT.Y416: 8,
        DXGI_FORMAT.NV12: 1,
        DXGI_FORMAT.P010: 2,
        DXGI_FORMAT.P016: 2,
        DXGI_FORMAT._420_OPAQUE: 0,
        DXGI_FORMAT.YUY2: 2,
        DXGI_FORMAT.Y210: 4,
        DXGI_FORMAT.Y216: 8,
        DXGI_FORMAT.NV11: 1,
        DXGI_FORMAT.AI44: 0,
        DXGI_FORMAT.IA44: 0,
        DXGI_FORMAT.P8: 0,
        DXGI_FORMAT.A8P8: 0,
        DXGI_FORMAT.B4G4R4A4_UNORM: 2,
        DXGI_FORMAT.P208: 0,
        DXGI_FORMAT.V208: 0,
        DXGI_FORMAT.V408: 0,
        DXGI_FORMAT.P208: 0,
        DXGI_FORMAT.V208: 0,
        DXGI_FORMAT.V408: 0,
        }
    return map[dxgi_format]


# https://github.com/Microsoft/DirectXTex/blob/main/DirectXTex/DDS.h

def int_to_fourcc(val: int) -> str:
    return struct.pack("<I", val).decode("ascii")

DDS_MAGIC = 0x20534444 # "DDS "
class PIXELFORMAT(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("fourCC", ctypes.c_uint32),
        ("RGBBitCount", ctypes.c_uint32),
        ("RBitMask", ctypes.c_uint32),
        ("GBitMask", ctypes.c_uint32),
        ("BBitMask", ctypes.c_uint32),
        ("ABitMask", ctypes.c_uint32)
    ]

    def Print(self):
        print(f"Size: {self.size}")
        print(f"Flags: {self.flags}")

        print(f"FourCC: {int_to_fourcc(self.fourCC)}")

        print(f"RGBBitCount: {self.RGBBitCount}")
        print(f"RBitMask: {self.RBitMask}")
        print(f"GBitMask: {self.GBitMask}")
        print(f"BBitMask: {self.BBitMask}")
        print(f"ABitMask: {self.ABitMask}")

def MAKE_FOURCC(a, b, c, d):
    return (ord(a) | (ord(b) << 8) | (ord(c) << 16) | (ord(d) << 24))

class RESOURCE_DIMENSION(Enum):
    TEXTURE1D = 2
    TEXTURE2D = 3
    TEXTURE3D = 4

class RESOURCE_MISC_FLAG(Enum):
    TEXTURECUBE = 0x4

class MISC_FLAGS2(Enum):
    ALPHA_MODE_MASK = 0x7

class ALPHA_MODE(Enum):
    UNKNOWN = 0
    STRAIGHT = 1
    PREMULTIPLIED = 2
    OPAQUE = 3
    CUSTOM = 4

class DDS_HEADER(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("pitch_or_linear_size", ctypes.c_uint32),
        ("depth", ctypes.c_uint32),
        ("mip_map_count", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32 * 11),
        ("ddspf", PIXELFORMAT),
        ("caps", ctypes.c_uint32),
        ("caps2", ctypes.c_uint32),
        ("caps3", ctypes.c_uint32),
        ("caps4", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32)
    ]

    def Print(self):
        print(f"Size: {self.size}")
        print(f"Flags: {self.flags}")
        print(f"Height: {self.height}")
        print(f"Width: {self.width}")
        print(f"pitch_or_linear_size: {self.pitch_or_linear_size}")
        print(f"Depth: {self.depth}")
        print(f"MipMapCount: {self.mip_map_count}")
        print(f"reserved1: {self.reserved1}")
        self.ddspf.Print()
        print(f"Caps: {self.caps}")
        print(f"Caps2: {self.caps2}")
        print(f"Caps3: {self.caps3}")
        print(f"Caps4: {self.caps4}")

class D3D10_RESOURCE_DIMENSION(Enum):
    UNKNOWN = 0
    BUFFER = 1
    TEXTURE1D = 2
    TEXTURE2D = 3
    TEXTURE3D = 4

class D3D10_RESOURCE_MISC_FLAG(Enum):
    TEXTURECUBE = 0x4


class DDS_HEADER_DXT10(ctypes.Structure):
    _fields_ = [
        ("dxgi_format", ctypes.c_uint32),
        ("resource_dimension", ctypes.c_uint32),
        ("misc_flag", ctypes.c_uint32),
        ("array_size", ctypes.c_uint32),
        ("misc_flags2", ctypes.c_uint32)
    ]

    def Print(self):
        print(f"dxgi_format: {DXGI_FORMAT(self.dxgi_format).name}")
        print(f"resource_dimension: {D3D10_RESOURCE_DIMENSION(self.resource_dimension).name}")
        print(f"misc_flag: {self.misc_flag}")
        print(f"array_size: {self.array_size}")
        print(f"misc_flags2: {self.misc_flags2}")

assert ctypes.sizeof(PIXELFORMAT) == 32, "DDS pixel format size mismatch"
assert ctypes.sizeof(DDS_HEADER) == 124, "DDS Header size mismatch"
assert ctypes.sizeof(DDS_HEADER_DXT10) == 20, "DDS DX10 Extended Header size mismatch"

class BufferWrapper:

    def __init__(self, buffer : np.ndarray, offset, size):
        self.buffer = buffer
        self.offset = offset
        self.size   = size
        assert self.buffer.dtype == np.uint8, "Buffer must be uint8"

    @property
    def ptr(self):
        return self.buffer.ctypes.data + self.offset
    
    def read(self, size):
        data = self.buffer[self.offset:self.offset+size]
        self.offset += size
        return data

    def read_ctype(self, ctype):
        obj = ctype()
        ctypes.memmove(ctypes.addressof(obj), self.ptr, ctypes.sizeof(ctype))
        self.offset += ctypes.sizeof(ctype)
        return obj

class DDSTexture:
    
    def __init__(self, path=None):
        if path is not None:
            with open(path, "rb") as f:
                self.buffer = np.frombuffer(f.read(), dtype=np.uint8)

        self.buf_ref        = BufferWrapper(self.buffer, 0, len(self.buffer))
        first_dword         = self.buf_ref.read(4)
        # print("--- ", self.buf_ref.offset)
        # print(int_to_fourcc(unpack("I", first_dword)[0]))
        assert unpack("I", first_dword)[0] == DDS_MAGIC, "Invalid DDS file"
        self.header         = self.buf_ref.read_ctype(DDS_HEADER)
        # print("--- ", self.buf_ref.offset)
        # self.header.Print()
        assert self.header.size == 124, "Only DDS_HEADER size 124 is supported"
        assert self.header.ddspf.size == 32, "Only DDS_PIXELFORMAT size 32 is supported"

        # assert self.header.ddspf.fourCC == MAKE_FOURCC('D', 'X', '1', '0'), "Only dx10 headers are supported"

        self.dx10_header = self.buf_ref.read_ctype(DDS_HEADER_DXT10)
        # print("--- ", self.buf_ref.offset)

        # self.dx10_header.Print()
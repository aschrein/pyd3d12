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

import os, sys
from pathlib import Path
import importlib.util
import psutil
import subprocess
import time
import ctypes
from .utils import *
from .dxc import *

Im = find_native_module("imgui")
native = find_native_module("native")

class ImVertex(ctypes.Structure):
    _fields_ = [
        ("pos", ctypes.c_float * 2),
        ("uv", ctypes.c_float * 2),
        ("col", ctypes.c_uint8 * 4)
    ]

class RootConstants(ctypes.Structure):
    _fields_ = [
        ("scale", ctypes.c_float * 2),
        ("translate", ctypes.c_float * 2)
    ]

assert ctypes.sizeof(ImVertex) == 20

class ImGuiContext:
    def __init__(self, device, hwnd):
        self.ctx = Im.CreateContext(hwnd)
        self.device = device
        self.vertex_buffer_size = 1 << 20
        self.vertex_buffer_cursor = 0
        self.index_buffer_size = 1 << 20
        self.index_buffer_cursor = 0
        self.vertex_buffer = self.device.CreateCommittedResource(
            heapProperties = native.D3D12_HEAP_PROPERTIES(
                Type = native.D3D12_HEAP_TYPE.CUSTOM,
                CPUPageProperty = native.D3D12_CPU_PAGE_PROPERTY.WRITE_COMBINE,
                MemoryPoolPreference = native.D3D12_MEMORY_POOL.L0,
                CreationNodeMask = 1,
                VisibleNodeMask = 1
            ),
            heapFlags = native.D3D12_HEAP_FLAGS.NONE,
            resourceDesc = native.D3D12_RESOURCE_DESC(
                Dimension = native.D3D12_RESOURCE_DIMENSION.BUFFER,
                Alignment = 0,
                Width = self.vertex_buffer_size,
                Height = 1,
                DepthOrArraySize = 1,
                MipLevels = 1,
                Format = native.DXGI_FORMAT.UNKNOWN,
                SampleDesc = native.DXGI_SAMPLE_DESC(
                    Count = 1,
                    Quality = 0
                ),
                Layout = native.D3D12_TEXTURE_LAYOUT.ROW_MAJOR,
                Flags = native.D3D12_RESOURCE_FLAGS.NONE
            ),
            initialState = native.D3D12_RESOURCE_STATES.VERTEX_AND_CONSTANT_BUFFER,
            optimizedClearValue = None
        )
        self.vertex_buffer_map = self.vertex_buffer.Map()
        self.index_buffer = self.device.CreateCommittedResource(
            heapProperties = native.D3D12_HEAP_PROPERTIES(
                Type = native.D3D12_HEAP_TYPE.CUSTOM,
                CPUPageProperty = native.D3D12_CPU_PAGE_PROPERTY.WRITE_COMBINE,
                MemoryPoolPreference = native.D3D12_MEMORY_POOL.L0,
                CreationNodeMask = 1,
                VisibleNodeMask = 1
            ),
            heapFlags = native.D3D12_HEAP_FLAGS.NONE,
            resourceDesc = native.D3D12_RESOURCE_DESC(
                Dimension = native.D3D12_RESOURCE_DIMENSION.BUFFER,
                Alignment = 0,
                Width = self.vertex_buffer_size,
                Height = 1,
                DepthOrArraySize = 1,
                MipLevels = 1,
                Format = native.DXGI_FORMAT.UNKNOWN,
                SampleDesc = native.DXGI_SAMPLE_DESC(
                    Count = 1,
                    Quality = 0
                ),
                Layout = native.D3D12_TEXTURE_LAYOUT.ROW_MAJOR,
                Flags = native.D3D12_RESOURCE_FLAGS.NONE
            ),
            initialState = native.D3D12_RESOURCE_STATES.INDEX_BUFFER,
            optimizedClearValue = None
        )
        self.index_buffer_map = self.index_buffer.Map()

        self.dxc_ctx = DXCContext()
        triangle_shader_text = """
//js
            #define ROOT_SIGNATURE_MACRO \
            "RootConstants(b0, num32BitConstants = 32), " \
            "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), " \

            struct VSInput {
                float2 position : POSITION;
                float2 uv : TEXCOORD;
                float4 color : COLOR;
            };
            struct PSInput {
                float4 position : SV_POSITION;
                float2 uv : TEXCOORD;
                float4 color : COLOR;
            };

            struct RootConstants {
                float2 scale;
                float2 translate;
            };

            ConstantBuffer<RootConstants> pc : register(b0);

            [RootSignature(ROOT_SIGNATURE_MACRO)]
            PSInput VSMain(VSInput input) {
                PSInput output;
                output.position = float4(pc.scale * input.position.xy + pc.translate, 0.0f, 1.0f);
                output.uv       = input.uv;
                output.color    = input.color;
                return output;
            }
            [RootSignature(ROOT_SIGNATURE_MACRO)]
            float4 PSMain(PSInput input) : SV_TARGET {
                return input.color;
            }
//!js
"""
        self.triangle_vertex_shader = self.dxc_ctx.compile_to_dxil(
            source = triangle_shader_text,
            args = "-E VSMain -T vs_6_5"
        )
        self.triangle_pixel_shader = self.dxc_ctx.compile_to_dxil(
            source = triangle_shader_text,
            args = "-E PSMain -T ps_6_5"
        )
        self.triangle_root_signature = self.device.CreateRootSignature(
            Bytes = self.triangle_vertex_shader
        )
        self.triangle_pipeline_state = self.device.CreateGraphicsPipelineState(
            native.D3D12_GRAPHICS_PIPELINE_STATE_DESC(
                RootSignature = self.triangle_root_signature,
                VS = native.D3D12_SHADER_BYTECODE(self.triangle_vertex_shader),
                PS = native.D3D12_SHADER_BYTECODE(self.triangle_pixel_shader),
                RasterizerState = native.D3D12_RASTERIZER_DESC(
                    FillMode = native.D3D12_FILL_MODE.SOLID,
                    CullMode = native.D3D12_CULL_MODE.NONE,
                    FrontCounterClockwise = False,
                    DepthBias = 0,
                    DepthBiasClamp = 0.0,
                    SlopeScaledDepthBias = 0.0,
                    DepthClipEnable = True,
                    MultisampleEnable = False,
                    AntialiasedLineEnable = False,
                    ForcedSampleCount = 0,
                    ConservativeRaster = native.D3D12_CONSERVATIVE_RASTERIZATION_MODE.OFF
                ),
                BlendState = native.D3D12_BLEND_DESC(
                    AlphaToCoverageEnable = False,
                    IndependentBlendEnable = False,
                    RenderTarget = [
                        native.D3D12_RENDER_TARGET_BLEND_DESC(
                            BlendEnable = False,
                            LogicOpEnable = False,
                            SrcBlend = native.D3D12_BLEND.ONE,
                            DestBlend = native.D3D12_BLEND.ZERO,
                            BlendOp = native.D3D12_BLEND_OP.ADD,
                            SrcBlendAlpha = native.D3D12_BLEND.ONE,
                            DestBlendAlpha = native.D3D12_BLEND.ZERO,
                            BlendOpAlpha = native.D3D12_BLEND_OP.ADD,
                            LogicOp = native.D3D12_LOGIC_OP.NOOP,
                            RenderTargetWriteMask = native.D3D12_COLOR_WRITE_ENABLE.ALL
                        )
                    ]
                ),
                DepthStencilState = native.D3D12_DEPTH_STENCIL_DESC(
                    DepthEnable = False,
                    DepthWriteMask = native.D3D12_DEPTH_WRITE_MASK.ALL,
                    DepthFunc = native.D3D12_COMPARISON_FUNC.LESS,
                    StencilEnable = False,
                    StencilReadMask = 0,
                    StencilWriteMask = 0,
                    FrontFace = native.D3D12_DEPTH_STENCILOP_DESC(
                        StencilFailOp = native.D3D12_STENCIL_OP.KEEP,
                        StencilDepthFailOp = native.D3D12_STENCIL_OP.KEEP,
                        StencilPassOp = native.D3D12_STENCIL_OP.KEEP,
                        StencilFunc = native.D3D12_COMPARISON_FUNC.ALWAYS
                    ),
                    BackFace = native.D3D12_DEPTH_STENCILOP_DESC(
                        StencilFailOp = native.D3D12_STENCIL_OP.KEEP,
                        StencilDepthFailOp = native.D3D12_STENCIL_OP.KEEP,
                        StencilPassOp = native.D3D12_STENCIL_OP.KEEP,
                        StencilFunc = native.D3D12_COMPARISON_FUNC.ALWAYS
                    )
                ),
                InputLayouts = [
                        native.D3D12_INPUT_ELEMENT_DESC(
                            SemanticName = "POSITION",
                            SemanticIndex = 0,
                            Format = native.DXGI_FORMAT.R32G32_FLOAT,
                            InputSlot = 0,
                            AlignedByteOffset = 0,
                            InputSlotClass = native.D3D12_INPUT_CLASSIFICATION.PER_VERTEX_DATA,
                            InstanceDataStepRate = 0
                        ),
                        native.D3D12_INPUT_ELEMENT_DESC(
                            SemanticName = "TEXCOORD",
                            SemanticIndex = 0,
                            Format = native.DXGI_FORMAT.R32G32_FLOAT,
                            InputSlot = 0,
                            AlignedByteOffset = 8,
                            InputSlotClass = native.D3D12_INPUT_CLASSIFICATION.PER_VERTEX_DATA,
                            InstanceDataStepRate = 0
                        ),
                        native.D3D12_INPUT_ELEMENT_DESC(
                            SemanticName = "COLOR",
                            SemanticIndex = 0,
                            Format = native.DXGI_FORMAT.R8G8B8A8_UNORM,
                            InputSlot = 0,
                            AlignedByteOffset = 16,
                            InputSlotClass = native.D3D12_INPUT_CLASSIFICATION.PER_VERTEX_DATA,
                            InstanceDataStepRate = 0
                        )
                    ],
                PrimitiveTopologyType = native.D3D12_PRIMITIVE_TOPOLOGY_TYPE.TRIANGLE,
                RTVFormats = [native.DXGI_FORMAT.R8G8B8A8_UNORM],
                DSVFormat = native.DXGI_FORMAT.UNKNOWN,
                SampleDesc = native.DXGI_SAMPLE_DESC(
                    Count = 1,
                    Quality = 0
                ),
                NodeMask = 0,
                Flags = native.D3D12_PIPELINE_STATE_FLAGS.NONE
            )
        )
        assert self.triangle_pipeline_state is not None


    def set_ctx(self):
        Im.SetCurrentContext(self.ctx)
        pass

    def render(self, cmd_list):

        Im.Render()

        draw_list = Im.GetDrawData()

        # print(f"draw_list.CmdListsCount: {draw_list.CmdListsCount}")

        cmd_list.SetPipelineState(self.triangle_pipeline_state)
        cmd_list.SetGraphicsRootSignature(self.triangle_root_signature)
   

        im_cmd_lists = draw_list.CmdLists
        for i in range(draw_list.CmdListsCount):
            imcmd_list = im_cmd_lists[i]
            cmd_buf = imcmd_list.CmdBuffer
            idx_buf_ptr = imcmd_list.IdxBufferPtr
            idx_buf_siz = imcmd_list.IdxBufferSize
            vtx_buf_ptr = imcmd_list.VtxBufferPtr
            vtx_buf_siz = imcmd_list.VtxBufferSize

            ctypes.memmove(self.vertex_buffer_map, vtx_buf_ptr, vtx_buf_siz)
            ctypes.memmove(self.index_buffer_map, idx_buf_ptr, idx_buf_siz)

            # print(f"len(cmd_buf): {len(cmd_buf)}")
            for cmd in cmd_buf:
                # print(f"cmd.ClipRect {cmd.ClipRect.x}, {cmd.ClipRect.y}, {cmd.ClipRect.z}, {cmd.ClipRect.w}")
                # print(f"cmd.ElemCount: {cmd.ElemCount}")
                cmd_list.IASetVertexBuffers(
                    StartSlot = 0,
                    Views = [
                        native.D3D12_VERTEX_BUFFER_VIEW(
                            BufferLocation = self.vertex_buffer.GetGPUVirtualAddress(),
                            SizeInBytes = vtx_buf_siz,
                            StrideInBytes = ctypes.sizeof(ImVertex)
                        )
                    ]
                )
                cmd_list.IASetIndexBuffer(native.D3D12_INDEX_BUFFER_VIEW(
                        BufferLocation = self.index_buffer.GetGPUVirtualAddress(),
                        SizeInBytes = idx_buf_siz,
                        Format = native.DXGI_FORMAT.R16_UINT
                    )
                )
                 
                cmd_list.IASetPrimitiveTopology(native.D3D12_PRIMITIVE_TOPOLOGY.TRIANGLELIST)
                pc = RootConstants()
                pc.scale[0] = 2.0 / draw_list.DisplaySize.x
                pc.scale[1] = 2.0 / draw_list.DisplaySize.y
                pc.translate[0] = -1.0
                pc.translate[1] = -1.0
                cmd_list.SetGraphicsRoot32BitConstants(
                    RootParameterIndex = 0,
                    Num32BitValuesToSet = 4,
                    SrcData = ctypes.addressof(pc),
                    DestOffsetIn32BitValues = 0
                )
                cmd_list.DrawIndexedInstanced(
                    IndexCountPerInstance = cmd.ElemCount,
                    InstanceCount = 1,
                    StartIndexLocation = cmd.IdxOffset,
                    BaseVertexLocation = cmd.VtxOffset,
                    StartInstanceLocation = 0
                )

                


        pass
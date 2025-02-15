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

import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import QTimer
import os, sys
from py.utils import *
from py.dxc import *
import ctypes

native = find_native_module("native")

class Vertex(ctypes.Structure):
    _fields_ = [
        ("position", ctypes.c_float * 3),
        ("color", ctypes.c_float * 4)
    ]

class MainWindow:
    def __init__(self):
        self.window = qtw.QMainWindow()
        self.window.setWindowTitle("Simple Window")
        self.window.setGeometry(100, 100, 800, 600)
        self.window.show()
        self.hwnd = int(self.window.winId())

        windth, height = native.GetWindowSize(self.hwnd)

        factory = native.IDXGIFactory()
        adapters = factory.EnumAdapters()
        print(f"Adapter: {adapters[0].GetDesc().Description}")
        self.factory = factory
        self.device = native.CreateDevice(adapters[0], native.D3D_FEATURE_LEVEL._11_0)
        self.command_queue = self.device.CreateCommandQueue(native.D3D12_COMMAND_QUEUE_DESC(
            Type = native.D3D12_COMMAND_LIST_TYPE.DIRECT,
            Priority = 0,
            Flags = native.D3D12_COMMAND_QUEUE_FLAGS.NONE,
            NodeMask = 0
        ))
        self.num_back_buffers = 3
        self.swapchain = self.factory.CreateSwapChain(
            self.command_queue,
            native.DXGI_SWAP_CHAIN_DESC(
                BufferDesc = native.DXGI_MODE_DESC(
                    Width = windth,
                    Height = height,
                    RefreshRate = native.DXGI_RATIONAL(
                        Numerator = 60,
                        Denominator = 1
                    ),
                    Format = native.DXGI_FORMAT.R8G8B8A8_UNORM,
                    ScanlineOrdering = native.DXGI_MODE_SCANLINE_ORDER.UNSPECIFIED,
                    Scaling = native.DXGI_MODE_SCALING.UNSPECIFIED
                ),
                SampleDesc = native.DXGI_SAMPLE_DESC(
                    Count = 1,
                    Quality = 0
                ),
                BufferUsage = native.DXGI_USAGE.RENDER_TARGET_OUTPUT | native.DXGI_USAGE.SHADER_INPUT,
                BufferCount = self.num_back_buffers,
                OutputWindow = self.hwnd,
                Windowed = True,
                SwapEffect = native.DXGI_SWAP_EFFECT.FLIP_DISCARD,
                Flags = native.DXGI_SWAP_CHAIN_FLAG.FRAME_LATENCY_WAITABLE_OBJECT
            )
        )
        self.frame_idx = 0

        self.fences = []
        self.cmd_allocs = []
        self.events = []
        for i in range(self.num_back_buffers):
            self.fences.append(self.device.CreateFence(0, native.D3D12_FENCE_FLAGS.NONE))
            self.events.append(native.Event())
            self.fences[i].SetEventOnCompletion(1, self.events[i])
            self.cmd_allocs.append(self.device.CreateCommandAllocator(native.D3D12_COMMAND_LIST_TYPE.DIRECT))
        
            # Mark fences as signaled
            self.fences[i].Signal(1)

        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.on_frame)
        self.render_timer.start(4)

        self.dxc_ctx = DXCContext()
        triangle_shader_text = """
//js
            #define ROOT_SIGNATURE_MACRO \
            "RootConstants(b0, num32BitConstants = 1), " \
            "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), " \

            struct VSInput {
                float3 position : POSITION;
                float4 color : COLOR;
            };
            struct PSInput {
                float4 position : SV_POSITION;
                float4 color : COLOR;
            };
            [RootSignature(ROOT_SIGNATURE_MACRO)]
            PSInput VSMain(VSInput input) {
                PSInput output;
                output.position = float4(input.position, 1.0f);
                output.color = input.color;
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
                    CullMode = native.D3D12_CULL_MODE.BACK,
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
                            Format = native.DXGI_FORMAT.R32G32B32_FLOAT,
                            InputSlot = 0,
                            AlignedByteOffset = 0,
                            InputSlotClass = native.D3D12_INPUT_CLASSIFICATION.PER_VERTEX_DATA,
                            InstanceDataStepRate = 0
                        ),
                        native.D3D12_INPUT_ELEMENT_DESC(
                            SemanticName = "COLOR",
                            SemanticIndex = 0,
                            Format = native.DXGI_FORMAT.R32G32B32A32_FLOAT,
                            InputSlot = 0,
                            AlignedByteOffset = 12,
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

       
        self.triangle_vertex_buffer = self.device.CreateCommittedResource(
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
                Width = ctypes.sizeof(Vertex) * 3,
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
            initialState = native.D3D12_RESOURCE_STATES.COPY_DEST,
            optimizedClearValue = None
        )

        mapped_ptr = self.triangle_vertex_buffer.Map()
        array = (Vertex * 3).from_address(mapped_ptr)
        array[0].position = (0.0, 0.5, 0.0)
        array[0].color = (1.0, 0.0, 0.0, 1.0)
        array[1].position = (0.5, -0.5, 0.0)
        array[1].color = (0.0, 1.0, 0.0, 1.0)
        array[2].position = (-0.5, -0.5, 0.0)
        array[2].color = (0.0, 0.0, 1.0, 1.0)
        self.triangle_vertex_buffer.Unmap()

        self.rtv_descritor_heap = self.device.CreateDescriptorHeap(
            native.D3D12_DESCRIPTOR_HEAP_DESC(
                Type = native.D3D12_DESCRIPTOR_HEAP_TYPE.RTV,
                NumDescriptors = 1024,
                Flags = native.D3D12_DESCRIPTOR_HEAP_FLAGS.NONE,
                NodeMask = 0
            )
        )
        self.rtv_descriptor_size = self.device.GetDescriptorHandleIncrementSize(native.D3D12_DESCRIPTOR_HEAP_TYPE.RTV)

        pass

    def on_frame(self):
        back_buffer_idx = self.swapchain.GetCurrentBackBufferIndex()
        # print(f"back_buffer_idx = {back_buffer_idx}")
       # if self.fences[back_buffer_idx].GetCompletedValue() != 1:
       #     # print(f"Waiting for fence {back_buffer_idx}")
        self.events[back_buffer_idx].Wait()
        self.events[back_buffer_idx].Reset()

        window_width, window_height = native.GetWindowSize(self.hwnd)
        back_buffer = self.swapchain.GetBuffer(back_buffer_idx)
        desc = back_buffer.GetDesc()
        if desc.Width != window_width or desc.Height != window_height:
            print(f"Resizing to {window_width}x{window_height}")

            del back_buffer # Release back buffer
            # Wait idle
            fence = self.device.CreateFence(0, native.D3D12_FENCE_FLAGS.NONE)
            self.command_queue.Signal(fence, 1)
            event = native.Event()
            fence.SetEventOnCompletion(1, event)
            event.Wait()

            for i in range(self.num_back_buffers):
                self.fences[i].Signal(1)
                self.fences[i].SetEventOnCompletion(1, self.events[i])

            self.swapchain.ResizeBuffers(
                self.num_back_buffers,
                window_width,
                window_height,
                native.DXGI_FORMAT.R8G8B8A8_UNORM,
                native.DXGI_SWAP_CHAIN_FLAG.FRAME_LATENCY_WAITABLE_OBJECT
            )
            back_buffer_idx = self.swapchain.GetCurrentBackBufferIndex()
            back_buffer = self.swapchain.GetBuffer(back_buffer_idx)


        self.cmd_allocs[back_buffer_idx].Reset()
        cmd_list = self.device.CreateCommandList(NodeMask=0, Type=native.D3D12_COMMAND_LIST_TYPE.DIRECT, Allocator=self.cmd_allocs[back_buffer_idx])
        cmd_list.SetPipelineState(self.triangle_pipeline_state)
        cmd_list.SetGraphicsRootSignature(self.triangle_root_signature)
   
        rtv_heap_offset_cpu = self.rtv_descritor_heap.GetCPUDescriptorHandleForHeapStart().ptr + back_buffer_idx * self.rtv_descriptor_size
        rtv_heap_offset_gpu = self.rtv_descritor_heap.GetGPUDescriptorHandleForHeapStart().ptr + back_buffer_idx * self.rtv_descriptor_size

        self.device.CreateRenderTargetView(
            Resource = back_buffer,
            Desc = native.D3D12_RENDER_TARGET_VIEW_DESC(
                Format = native.DXGI_FORMAT.R8G8B8A8_UNORM,
                Texture2D = native.D3D12_TEX2D_RTV(
                    MipSlice = 0,
                    PlaneSlice = 0
                )
            ),
            DestDescriptor = native.D3D12_CPU_DESCRIPTOR_HANDLE(rtv_heap_offset_cpu)
        )

        cmd_list.RSSetViewports(Viewports=[native.D3D12_VIEWPORT(
            TopLeftX = 0,
            TopLeftY = 0,
            Width = window_width,
            Height = window_height,
            MinDepth = 0.0,
            MaxDepth = 1.0
        )])
        cmd_list.RSSetScissorRects(Rects=[native.D3D12_RECT(
            left = 0,
            top = 0,
            right = window_width,
            bottom = window_height
        )])
        
        cmd_list.ResourceBarrier([
            native.D3D12_RESOURCE_BARRIER(
                Transition = native.D3D12_RESOURCE_TRANSITION_BARRIER(
                    Resource = back_buffer,
                    Subresource = 0,
                    StateBefore = native.D3D12_RESOURCE_STATES.PRESENT,
                    StateAfter = native.D3D12_RESOURCE_STATES.RENDER_TARGET
                )
            )
        ])
        cmd_list.OMSetRenderTargets(RenderTargetDescriptors=[native.D3D12_CPU_DESCRIPTOR_HANDLE(rtv_heap_offset_cpu)])
        cmd_list.ClearRenderTargetView(View=native.D3D12_CPU_DESCRIPTOR_HANDLE(rtv_heap_offset_cpu), Color=(0.5, 0.5, 0.5, 1.0), Rects=None)
        cmd_list.IASetVertexBuffers(
            StartSlot = 0,
            Views = [
                native.D3D12_VERTEX_BUFFER_VIEW(
                    BufferLocation = self.triangle_vertex_buffer.GetGPUVirtualAddress(),
                    SizeInBytes = ctypes.sizeof(Vertex) * 3,
                    StrideInBytes = ctypes.sizeof(Vertex)
                )
            ]
        )
        cmd_list.IASetPrimitiveTopology(native.D3D12_PRIMITIVE_TOPOLOGY.TRIANGLELIST)
        cmd_list.DrawInstanced(3, 1, 0, 0)

        
        cmd_list.ResourceBarrier([
            native.D3D12_RESOURCE_BARRIER(
                Transition = native.D3D12_RESOURCE_TRANSITION_BARRIER(
                    Resource = back_buffer,
                    Subresource = 0,
                    StateBefore = native.D3D12_RESOURCE_STATES.RENDER_TARGET,
                    StateAfter = native.D3D12_RESOURCE_STATES.PRESENT
                )
            )
        ])

        cmd_list.Close()
        self.command_queue.ExecuteCommandLists([cmd_list])

        # print(f"self.fences[back_buffer_idx].GetCompletedValue() {self.fences[back_buffer_idx].GetCompletedValue()}")
        # assert self.fences[back_buffer_idx].GetCompletedValue() == 1
        self.fences[back_buffer_idx].Signal(0)

        self.swapchain.Present(0, 0)

        self.command_queue.Signal(self.fences[back_buffer_idx], 1)
        self.fences[back_buffer_idx].SetEventOnCompletion(1, self.events[back_buffer_idx])

        # print(f"Frame: {self.frame_idx}")
        self.frame_idx = self.frame_idx + 1

    def run(self):
        pass

if __name__ == "__main__":
    launch_debugviewpp()

    debug = native.ID3D12Debug()
    debug.EnableDebugLayer()

    app = qtw.QApplication([])
    window = MainWindow()
    window.run()
    app.exec_()
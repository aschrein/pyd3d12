# MIT License
# Copyright (c) 2025 Anton Schreiner

import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, QObject, QTimer, QEvent
from PyQt5.QtGui import QSurfaceFormat
import os, sys
from py.utils import *
from py.dxc import *
import ctypes
import argparse

args = argparse.ArgumentParser()
args.add_argument("--build", type=str, default="Release")
args.add_argument("--wait_for_debugger_present", action="store_true")
args.add_argument("--load_rdoc", action="store_true")
args = args.parse_args()
set_build_type(args.build)

from py.d3d12 import *

if args.wait_for_debugger_present:
    print("Waiting for debugger to attach...")
    while not native.IsDebuggerPresent():
        pass

from py.imgui import *
from py.rdoc import *

if args.load_rdoc:
    if find_rdoc() is not None:
        rdoc_load()

class Vertex(ctypes.Structure):
    _fields_ = [
        ("position", ctypes.c_float * 3),
        ("color", ctypes.c_float * 4)
    ]

class ChildWidget(qtw.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("child_widget")
        self.setStyleSheet("background-color: lightgray;")  # for visibility
        self.imgui_ctx = None # set later

    def mouseMoveEvent(self, event):
        # print("Child widget mouse move:", event.pos())
        self.imgui_ctx.ctx.OnMouseMotion(event.pos().x(), event.pos().y())
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        # if event.button() == Qt.LeftButton:
        #     print("Child widget left button pressed at:", event.pos())
        key = 0
        if event.button() == Qt.LeftButton: key = 0
        if event.button() == Qt.RightButton: key = 1
        if event.button() == Qt.MiddleButton: key = 2

        self.imgui_ctx.ctx.OnMouseMotion(event.pos().x(), event.pos().y())
        self.imgui_ctx.ctx.OnMousePress(key)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        # if event.button() == Qt.LeftButton:
        #     print("Child widget left button released at:", event.pos())
        key = 0
        if event.button() == Qt.LeftButton: key = 0
        if event.button() == Qt.RightButton: key = 1
        if event.button() == Qt.MiddleButton: key = 2

        self.imgui_ctx.ctx.OnMouseMotion(event.pos().x(), event.pos().y())
        self.imgui_ctx.ctx.OnMouseRelease(key)
        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        # print("Mouse entered the child widget")
        super().enterEvent(event)

    def leaveEvent(self, event):
        # print("Mouse left the child widget")
        super().leaveEvent(event)
    
    def keyPressEvent(self, event):
        # print("Key pressed:", event.key())
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        # print("Key released:", event.key())
        super().keyReleaseEvent(event)

class MainWindow:
    def __init__(self):
        self.window = qtw.QMainWindow()
        self.child = ChildWidget(self.window)
        self.window.setCentralWidget(self.child)
        self.child.setGeometry(50, 50, 300, 200)

        self.window.setWindowTitle("Simple Window")
        self.window.setGeometry(100, 100, 800, 600)
        self.window.show()
        self.hwnd = int(self.child.winId())

        windth, height = native.GetWindowSize(self.hwnd)

        factory = native.IDXGIFactory()
        adapters = factory.EnumAdapters()
        print(f"Adapter: {adapters[0].GetDesc().Description}")
        self.factory = factory
        self.device = native.CreateDevice(adapters[0], native.D3D_FEATURE_LEVEL._11_0)
        self.imgui_ctx = ImGuiContext(self.device, self.hwnd)
        self.child.imgui_ctx = self.imgui_ctx

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
                        Numerator = 0,
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

        self.dxc_ctx = DXCContext()
        triangle_shader_text = """
//js
            #define ROOT_SIGNATURE_MACRO \
            "DescriptorTable(" \
            "SRV(t0, NumDescriptors = 1, flags = DESCRIPTORS_VOLATILE) " \
            "), " \
            "StaticSampler(s0, " \
                "Filter = FILTER_MIN_MAG_MIP_LINEAR, " \
                "AddressU = TEXTURE_ADDRESS_WRAP, " \
                "AddressV = TEXTURE_ADDRESS_WRAP, " \
                "AddressW = TEXTURE_ADDRESS_WRAP), " \
            "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), " \

            struct VSInput {
                float3 position : POSITION;
                float4 color : COLOR;
            };
            struct PSInput {
                float4 position : SV_POSITION;
                float4 color : COLOR;
            };

            Texture2D<float4> t0 : register(t0);
            SamplerState s0 : register(s0);

            [RootSignature(ROOT_SIGNATURE_MACRO)]
            PSInput VSMain(VSInput input) {
                PSInput output;
                output.position = float4(input.position, 1.0f);
                output.color = input.color;
                return output;
            }
            [RootSignature(ROOT_SIGNATURE_MACRO)]
            float4 PSMain(PSInput input) : SV_TARGET {
                return pow(t0.SampleLevel(s0, input.color.xy, input.color.x * float(8.0)), float(1.0 / 1.0));
                // return t0[abs(input.color.xy) * 512];
                // return input.color;
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
        self.triangle_pso = self.device.CreateGraphicsPipelineState(
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
                            BlendEnable = True,
                            LogicOpEnable = False,
                            SrcBlend = native.D3D12_BLEND.SRC_ALPHA,
                            DestBlend = native.D3D12_BLEND.INV_SRC_ALPHA,
                            BlendOp = native.D3D12_BLEND_OP.ADD,
                            SrcBlendAlpha = native.D3D12_BLEND.SRC_ALPHA,
                            DestBlendAlpha = native.D3D12_BLEND.INV_SRC_ALPHA,
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
        assert self.triangle_pso is not None

       
        self.triangle_vertex_buffer = make_write_combined_buffer(self.device, size=ctypes.sizeof(Vertex) * 3, state=native.D3D12_RESOURCE_STATES.VERTEX_AND_CONSTANT_BUFFER, name="Triangle Vertex Buffer")

        mapped_ptr = self.triangle_vertex_buffer.Map()
        array = (Vertex * 3).from_address(mapped_ptr)
        array[0].position       = (-1.0, -1.0, 0.0)
        array[0].color          = (0.0, 2.0, 0.0, 1.0)
        array[1].position       = (-1.0, 3.0, 0.0)
        array[1].color          = (0.0, .0, 0.0, 1.0)
        array[2].position       = (3.0, -1.0, 0.0)
        array[2].color          = (2.0, 2.0, 1.0, 1.0)
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

        asset_folder = find_file_or_folder("assets")
        assert asset_folder is not None

        dds_file = DDSTexture(asset_folder / "mandrill.dds")

        self.texture = make_texture_from_dds(self.device, dds_file)

        self.cbv_srv_uav_heap = CBV_SRV_UAV_DescriptorHeap(self.device, 1024)

        cpu_handle, gpu_handle = self.cbv_srv_uav_heap.get_next_descriptor_handle()

        self.device.CreateShaderResourceView(
            Resource = self.texture,
            Desc = native.D3D12_SHADER_RESOURCE_VIEW_DESC(
                Format = self.texture.GetDesc().Format,
                Shader4ComponentMapping = native.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Texture2D = native.D3D12_TEX2D_SRV(
                    MipLevels = self.texture.GetDesc().MipLevels,
                    MostDetailedMip = 0,
                    PlaneSlice = 0,
                    ResourceMinLODClamp = 0.0
                )
            ),
            DestDescriptor = cpu_handle
        )
        self.texture_gpu_descritpor = gpu_handle

        self.last_time = time.time()

        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.on_frame)
        self.render_timer.start(0)

        pass

    def on_frame(self):

        self.cur_time = time.time()
        dt = self.cur_time - self.last_time
        self.last_time = self.cur_time
        # print(f"dt = {dt}")

        # !Imgui
        window_width, window_height = native.GetWindowSize(self.hwnd)
        self.imgui_ctx.set_ctx()
        self.imgui_ctx.ctx.SetDisplaySize(window_width, window_height)
        Im.NewFrame()
        # Im.SetNextWindowSize(Im.Vec2(200, 100))
        Im.Begin("Hello, world!")
        Im.End()
        Im.Begin("Second Window!")
        Im.End()
       
        back_buffer_idx = self.swapchain.GetCurrentBackBufferIndex()
        # print(f"back_buffer_idx = {back_buffer_idx}")
        # if self.fences[back_buffer_idx].GetCompletedValue() != 1:
        #     # print(f"Waiting for fence {back_buffer_idx}")

        # !Wait for fence
        self.events[back_buffer_idx].Wait()
        self.events[back_buffer_idx].Reset()
        self.fences[back_buffer_idx].Signal(0)

        
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
        
        cmd_list.SetPipelineState(self.triangle_pso)
        cmd_list.SetGraphicsRootSignature(self.triangle_root_signature)
        cmd_list.SetDescriptorHeaps([self.cbv_srv_uav_heap.heap])
        cmd_list.SetGraphicsRootDescriptorTable(
            RootParameterIndex = 0,
            BaseDescriptor = self.texture_gpu_descritpor
        )
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

        self.imgui_ctx.render(cmd_list)

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
        
        self.command_queue.Signal(self.fences[back_buffer_idx], 1)
        self.fences[back_buffer_idx].SetEventOnCompletion(1, self.events[back_buffer_idx])

        self.swapchain.Present(0, 0)

        # print(f"Frame: {self.frame_idx}")
        self.frame_idx = self.frame_idx + 1

    def run(self):
        pass

launch_debugviewpp()

debug = native.ID3D12Debug()
debug.EnableDebugLayer()

fmt = QSurfaceFormat()
fmt.setSwapInterval(0)
QSurfaceFormat.setDefaultFormat(fmt)

app = qtw.QApplication([])
window = MainWindow()
window.run()
app.exec_()
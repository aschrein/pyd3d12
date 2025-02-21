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
import ctypes
import argparse
from py.utils import *
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import Qt, QObject, QTimer, QEvent
from PyQt5.QtGui import QSurfaceFormat

args = argparse.ArgumentParser()
args.add_argument("--build", type=str, default="Release")
args.add_argument("--wait_for_debugger_present", action="store_true")
args.add_argument("--load_rdoc", action="store_true")
args.add_argument("--enable_shader_clock", action="store_true")
args.add_argument("--load_pix", action="store_true")
args.add_argument("--gltf_scene_path", type=str )
args = args.parse_args()
set_build_type(args.build)

from py.dxc import *
from py.rdoc import *
from py.d3d12 import *
from py.pix import *
from py.gltf import *
from py.imgui import *
from py.linalg import *
from py.blue_noise import *

launch_debugviewpp()

if args.load_pix:
    pix.PIXLoadLatestWinPixGpuCapturerLibrary()
    # pix.PIXBeginCapture()

if args.load_rdoc and  find_rdoc() is not None:
    rdoc_load()

native = find_native_module("native")

debug = native.ID3D12Debug()
debug.EnableDebugLayer()
# debug.SetEnableGPUBasedValidation(True)

class Vertex(ctypes.Structure):
    _fields_ = [
        ("position", ctypes.c_float * 3),
        ("color", ctypes.c_float * 4)
    ]

class GeometryDesc(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint),
        ("position_offset_dwords", ctypes.c_uint),
        ("normals_offset_dwords", ctypes.c_uint),
        ("texcoord_offset_dwords", ctypes.c_uint),
        ("tangent_offset_dwords", ctypes.c_uint),
        ("indices_offset_dwords", ctypes.c_uint)
    ]

class CBuffer(ctypes.Structure):
    _fields_ = [
        ("frustum_x", ctypes.c_float * 3),
        ("aspect", ctypes.c_float),
        ("frustum_y", ctypes.c_float * 3),
        ("half_fov_tan", ctypes.c_float),
        ("frustum_z", ctypes.c_float * 3),
        ("pad_0", ctypes.c_float),
        ("camera_pos", ctypes.c_float * 3),
        ("frame_idx", ctypes.c_uint)

    ]

camera = Camera()
# camera.y_is_up = False

import json

try:
    with open(get_or_create_tmp_folder() / "camera.json", "r") as f:
        camera.from_json(json.load(f))
except:
    pass

# register on close
def on_close():
    with open(get_or_create_tmp_folder() / "camera.json", "w") as f:
        json.dump(camera.to_json(), f)


key_press_map = {}

class ChildWidget(qtw.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("child_widget")
        self.setStyleSheet("background-color: lightgray;")  # for visibility
        self.imgui_ctx = None # set later

        self.mouse_pressed = [False, False, False]
        self.mouse_last = (0, 0)
        self.mouse_in_window = False

    def mouseMoveEvent(self, event):
        # print("Child widget mouse move:", event.pos())
        self.imgui_ctx.ctx.OnMouseMotion(event.pos().x(), event.pos().y())

        mouse_delta     = (event.pos().x() - self.mouse_last[0], event.pos().y() - self.mouse_last[1])
        self.mouse_last = (event.pos().x(), event.pos().y())
        if self.mouse_pressed[0]:
            camera.phi   += mouse_delta[0] * 0.01
            camera.theta -= mouse_delta[1] * 0.01

        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        # if event.button() == Qt.LeftButton:
        #     print("Child widget left button pressed at:", event.pos())
        key = 0
        if event.button() == Qt.LeftButton: key = 0
        if event.button() == Qt.RightButton: key = 1
        if event.button() == Qt.MiddleButton: key = 2
        self.mouse_pressed[key] = True
        self.mouse_last = (event.pos().x(), event.pos().y())
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
        self.mouse_pressed[key] = False
        self.mouse_last = (event.pos().x(), event.pos().y())
        self.imgui_ctx.ctx.OnMouseMotion(event.pos().x(), event.pos().y())
        self.imgui_ctx.ctx.OnMouseRelease(key)
        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        # print("Mouse entered the child widget")
        self.mouse_in_window = True
        super().enterEvent(event)

    def leaveEvent(self, event):
        # print("Mouse left the child widget")
        self.mouse_in_window = False
        super().leaveEvent(event)
    
    def keyPressEvent(self, event):
        # print("Key pressed:", event.key())

        key_press_map[event.key()] = True

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        
        key_press_map[event.key()] = False

        # print("Key released:", event.key())
        super().keyReleaseEvent(event)

    def closeEvent(self, event):
        self.render_timer.stop()  # Stop further rendering
        on_close()
        print_red("Closing window")
        event.accept()


class MainWindow:
    def __init__(self):
        self.window     = qtw.QMainWindow()
        self.child      = ChildWidget(self.window)
        self.window.setCentralWidget(self.child)
        self.child.setGeometry(50, 50, 1024, 1024)
        self.child.setFocusPolicy(Qt.StrongFocus)

        self.window.setWindowTitle("Simple Window")
        self.window.setGeometry(100, 100, 1024, 1024)
        self.window.show()
        self.hwnd = int(self.child.winId())

        windth, height = native.GetWindowSize(self.hwnd)

        factory         = native.IDXGIFactory()
        adapters        = factory.EnumAdapters()
        print(f"Adapter: {adapters[0].GetDesc().Description}")
        self.factory = factory
        self.ags_context = native.AGSContext()
        if self.ags_context.IsValid():
            print_green("AGS Context is valid")
            self.device = self.ags_context.CreateDevice(adapters[0], native.D3D_FEATURE_LEVEL._11_0)
        else:
            self.device = native.CreateDevice(adapters[0], native.D3D_FEATURE_LEVEL._11_0)
        
        self.imgui_ctx = ImGuiContext(self.device, self.hwnd)
        self.child.imgui_ctx = self.imgui_ctx

        self.cbuffer_wb = make_write_combined_buffer(self.device, size=ctypes.sizeof(CBuffer) * 3, state=native.D3D12_RESOURCE_STATES.VERTEX_AND_CONSTANT_BUFFER, name="CBuffer")
        cbuffer_ptr = self.cbuffer_wb.Map()
        self.cbuffer_wb_arr = (CBuffer * 3).from_address(cbuffer_ptr)
        
        # Create blue noise sampler buffers
        sobol_ptr                   = blue_noise._128x128_2d2d2d2d_256spp_sobol_ptr()
        sobol_size_bytes            = blue_noise._128x128_2d2d2d2d_256spp_sobol_size_bytes()
        scrambling_tile_ptr         = blue_noise._128x128_2d2d2d2d_256spp_scrambling_tile_ptr()
        scrambling_tile_size_bytes  = blue_noise._128x128_2d2d2d2d_256spp_scrambling_tile_size_bytes()
        ranking_tile_ptr            = blue_noise._128x128_2d2d2d2d_256spp_ranking_tile_ptr()
        ranking_tile_size_bytes     = blue_noise._128x128_2d2d2d2d_256spp_ranking_tile_size_bytes()

        self.sobol_buffer = make_uav_buffer(self.device,
                                            size=sobol_size_bytes,
                                            state=native.D3D12_RESOURCE_STATES.UNORDERED_ACCESS,
                                            name="sobol_buffer",
                                            initial_raw_ptr=sobol_ptr)
    
        self.scrambling_tile_buffer = make_uav_buffer(self.device,
                                            size=scrambling_tile_size_bytes,
                                            state=native.D3D12_RESOURCE_STATES.UNORDERED_ACCESS,
                                            name="scrambling_tile_buffer",
                                            initial_raw_ptr=scrambling_tile_ptr)
        
        self.ranking_tile_buffer = make_uav_buffer(self.device,
                                            size=ranking_tile_size_bytes,
                                            state=native.D3D12_RESOURCE_STATES.UNORDERED_ACCESS,
                                            name="ranking_tile_buffer",
                                            initial_raw_ptr=ranking_tile_ptr)



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
        self.frame_idx      = 0
        self.fences         = []
        self.cmd_allocs     = []
        self.events         = []
        for i in range(self.num_back_buffers):
            self.fences.append(self.device.CreateFence(0, native.D3D12_FENCE_FLAGS.NONE))
            self.events.append(native.Event())
            self.fences[i].SetEventOnCompletion(1, self.events[i])
            self.cmd_allocs.append(self.device.CreateCommandAllocator(native.D3D12_COMMAND_LIST_TYPE.DIRECT))
        
            # Mark fences as signaled
            self.fences[i].Signal(1)

        self.dxc_ctx = DXCContext()
        self.dxc_ctx.add_include_path(get_third_party_folder() / "ags_lib/hlsl")
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
                // return pow(t0.SampleLevel(s0, input.color.xy, input.color.x * float(8.0)), float(1.0 / 1.0));
                return pow(t0.SampleLevel(s0, input.color.xy, 0.0f), float(1.0 / 1.0));
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

        amd_ags_define = ""
        if self.ags_context.IsValid() and args.enable_shader_clock:
            amd_ags_define = "-D AMD_AGS_ENABLED"
        
        hlsl_folder = Path(__file__).parent / "hlsl"
        assert hlsl_folder.exists()

        bytecode = self.dxc_ctx.compile_to_dxil(
            source = hlsl_folder / "simple_rt_mlp.hlsl",
            args = "-E main -T cs_6_5 " + amd_ags_define,
        )
        assert bytecode is not None

        self.rt_signature = self.device.CreateRootSignature(
            Bytes = bytecode
        )
        assert self.rt_signature is not None

        signature = self.rt_signature
        device = self.device
        dxc_ctx = self.dxc_ctx
        Main_pso = device.CreateComputePipelineState(native.D3D12_COMPUTE_PIPELINE_STATE_DESC(
            RootSignature = signature,
            CS = native.D3D12_SHADER_BYTECODE(dxc_ctx.compile_to_dxil(source = hlsl_folder / "simple_rt_mlp.hlsl", args = "-E Main -T cs_6_5",)),
            NodeMask = 0,
            CachedPSO = None,
            Flags = native.D3D12_PIPELINE_STATE_FLAGS.NONE
        ))
        Backwards_pso = device.CreateComputePipelineState(native.D3D12_COMPUTE_PIPELINE_STATE_DESC(
            RootSignature = signature,
            CS = native.D3D12_SHADER_BYTECODE(dxc_ctx.compile_to_dxil(source = hlsl_folder / "simple_rt_mlp.hlsl", args = "-E Backward -T cs_6_5",)),
            NodeMask = 0,
            CachedPSO = None,
            Flags = native.D3D12_PIPELINE_STATE_FLAGS.NONE
        ))
        Inference_pso = device.CreateComputePipelineState(native.D3D12_COMPUTE_PIPELINE_STATE_DESC(
            RootSignature = signature,
            CS = native.D3D12_SHADER_BYTECODE(dxc_ctx.compile_to_dxil(source = hlsl_folder / "simple_rt_mlp.hlsl", args = "-E InferencePass -T cs_6_5",)),
            NodeMask = 0,
            CachedPSO = None,
            Flags = native.D3D12_PIPELINE_STATE_FLAGS.NONE
        ))
        Initialize_pso = device.CreateComputePipelineState(native.D3D12_COMPUTE_PIPELINE_STATE_DESC(
            RootSignature = signature,
            CS = native.D3D12_SHADER_BYTECODE(dxc_ctx.compile_to_dxil(source = hlsl_folder / "simple_rt_mlp.hlsl", args = "-E Initialize -T cs_6_5",)),
            NodeMask = 0,
            CachedPSO = None,
            Flags = native.D3D12_PIPELINE_STATE_FLAGS.NONE
        ))
        self.Main_pso = Main_pso
        self.Backwards_pso = Backwards_pso
        self.Inference_pso = Inference_pso
        self.Initialize_pso = Initialize_pso


        pso_desc = native.D3D12_COMPUTE_PIPELINE_STATE_DESC(
            RootSignature   = self.rt_signature,
            CS              = native.D3D12_SHADER_BYTECODE(bytecode),
            NodeMask        = 0,
            CachedPSO       = None,
            Flags           = native.D3D12_PIPELINE_STATE_FLAGS.NONE
        )
        self.rt_pso = self.device.CreateComputePipelineState(pso_desc)

        assert self.rt_pso is not None

        self.gltf_scene = GLTFScene(args.gltf_scene_path)

        width, height = 512, 512

        cmd_queue = self.device.CreateCommandQueue(native.D3D12_COMMAND_QUEUE_DESC(
            Type = native.D3D12_COMMAND_LIST_TYPE.DIRECT,
            Priority = 0,
            Flags = native.D3D12_COMMAND_QUEUE_FLAGS.NONE,
            NodeMask = 0
        ))
       
        cmd_alloc = self.device.CreateCommandAllocator(native.D3D12_COMMAND_LIST_TYPE.DIRECT)
        
        geometry_descs  = []

        total_buffer_size = 0

        for mesh in self.gltf_scene.meshes:
            for primitive in mesh.primitives:
                cube_indices    = primitive.indices
                cube_vertices   = primitive.attributes["POSITION"]
                normals         = primitive.attributes.get("NORMAL", None)
                texcoords       = primitive.attributes.get("TEXCOORD_0", None)
                tangents        = primitive.attributes.get("TANGENT", None)
                total_buffer_size += cube_vertices.size * ctypes.sizeof(GltfFloat3)
                total_buffer_size  = (total_buffer_size + 255) & ~255

                if normals is not None:
                    total_buffer_size += normals.size * ctypes.sizeof(GltfFloat3)
                    total_buffer_size  = (total_buffer_size + 255) & ~255
                
                if texcoords is not None:
                    total_buffer_size += texcoords.size * ctypes.sizeof(GltfFloat2)
                    total_buffer_size  = (total_buffer_size + 255) & ~255
                
                if tangents is not None:
                    total_buffer_size += tangents.size * ctypes.sizeof(GltfFloat4)
                    total_buffer_size  = (total_buffer_size + 255) & ~255

                total_buffer_size += cube_indices.size * ctypes.sizeof(ctypes.c_uint32)
                total_buffer_size  = (total_buffer_size + 255) & ~255
            
        mega_buffer = make_write_combined_buffer(self.device, size=total_buffer_size)
        mapped_ptr = mega_buffer.Map()

        cur_buffer_offset = 0

        self.shader_geometry_descs = []

        for mesh in self.gltf_scene.meshes:
            for primitive in mesh.primitives:
                cube_indices    = primitive.indices
                assert cube_indices.dtype == GltfUint32 or cube_indices.dtype == GltfUint16, f"cube_indices: {cube_indices}"
                cube_vertices   = primitive.attributes["POSITION"]
                normals         = primitive.attributes.get("NORMAL", None)
                texcoords       = primitive.attributes.get("TEXCOORD_0", None)
                tangents        = primitive.attributes.get("TANGENT", None)

                index_offset_dwords        = 0xffffffff
                vertex_offset_dwords       = 0xffffffff
                normal_offset_dwords       = 0xffffffff
                texcoord_offset_dwords     = 0xffffffff
                tangent_offset_dwords      = 0xffffffff
                indices_offset_dwords      = 0xffffffff


                vertex_offset_dwords = cur_buffer_offset // 4
                ctypes.memmove(mapped_ptr + cur_buffer_offset, cube_vertices.ctypes.data, cube_vertices.size * ctypes.sizeof(GltfFloat3))
                cur_buffer_offset += cube_vertices.size * ctypes.sizeof(GltfFloat3)
                cur_buffer_offset  = (cur_buffer_offset + 255) & ~255 # align to 256 bytes
                
                assert normals is not None

                if normals is not None:
                    normal_offset_dwords = cur_buffer_offset // 4
                    ctypes.memmove(mapped_ptr + cur_buffer_offset, normals.ctypes.data, normals.size * ctypes.sizeof(GltfFloat3))
                    cur_buffer_offset += normals.size * ctypes.sizeof(GltfFloat3)
                    cur_buffer_offset  = (cur_buffer_offset + 255) & ~255 # align to 256 bytes

                if texcoords is not None:
                    texcoord_offset_dwords = cur_buffer_offset // 4
                    ctypes.memmove(mapped_ptr + cur_buffer_offset, texcoords.ctypes.data, texcoords.size * ctypes.sizeof(GltfFloat2))
                    cur_buffer_offset += texcoords.size * ctypes.sizeof(GltfFloat2)
                    cur_buffer_offset  = (cur_buffer_offset + 255) & ~255

                if tangents is not None:
                    tangent_offset_dwords = cur_buffer_offset // 4
                    ctypes.memmove(mapped_ptr + cur_buffer_offset, tangents.ctypes.data, tangents.size * ctypes.sizeof(GltfFloat4))
                    cur_buffer_offset += tangents.size * ctypes.sizeof(GltfFloat4)
                    cur_buffer_offset  = (cur_buffer_offset + 255) & ~255

                indices_offset_dwords = cur_buffer_offset // 4
                ctypes.memmove(mapped_ptr + cur_buffer_offset, cube_indices.ctypes.data, cube_indices.size * cube_indices.itemsize)
                cur_buffer_offset += cube_indices.size * cube_indices.itemsize
                cur_buffer_offset  = (cur_buffer_offset + 255) & ~255 # align to 256 bytes
                
                flags = 0
                if cube_indices.dtype == GltfUint16:
                    flags |= 1

                self.shader_geometry_descs.append(
                    GeometryDesc(
                        flags = flags,
                        position_offset_dwords = vertex_offset_dwords,
                        normals_offset_dwords = normal_offset_dwords,
                        texcoord_offset_dwords = texcoord_offset_dwords,
                        tangent_offset_dwords = tangent_offset_dwords,
                        indices_offset_dwords = indices_offset_dwords
                    )
                )

                geometry_descs.append(native.D3D12_RAYTRACING_GEOMETRY_DESC(
                    Flags = native.D3D12_RAYTRACING_GEOMETRY_FLAGS.OPAQUE,
                    Triangles = native.D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC(
                        Transform3x4               = 0,
                        IndexFormat                = native.DXGI_FORMAT.R32_UINT if cube_indices.dtype == GltfUint32 else native.DXGI_FORMAT.R16_UINT,
                        VertexFormat               = native.DXGI_FORMAT.R32G32B32_FLOAT,
                        IndexCount                 = cube_indices.size,
                        VertexCount                = cube_vertices.size,
                        IndexBuffer                = mega_buffer.GetGPUVirtualAddress() + indices_offset_dwords * 4,
                        VertexBuffer               = native.D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE(
                            StartAddress             = mega_buffer.GetGPUVirtualAddress() + vertex_offset_dwords * 4,
                            StrideInBytes            = ctypes.sizeof(GltfFloat3)
                        ),
                    )
                ))


        self.shader_geometry_descs_buffer = make_write_combined_buffer(self.device, size=len(self.shader_geometry_descs) * ctypes.sizeof(GeometryDesc))
        mapped_ptr = self.shader_geometry_descs_buffer.Map()
        array = (GeometryDesc * len(self.shader_geometry_descs)).from_address(mapped_ptr)
        for i in range(len(self.shader_geometry_descs)):
            array[i] = self.shader_geometry_descs[i]
        self.shader_geometry_descs_buffer.Unmap()

        as_inputs = native.D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS(
            Flags = native.D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS.PREFER_FAST_TRACE | native.D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS.ALLOW_UPDATE,
            GeometryDescs = geometry_descs
        )

        prebuild_info = self.device.GetRaytracingAccelerationStructurePrebuildInfo(as_inputs)
        print(f"ResultDataMaxSizeInBytes: {prebuild_info.ResultDataMaxSizeInBytes}")
        print(f"ScratchDataSizeInBytes: {prebuild_info.ScratchDataSizeInBytes}")
        print(f"UpdateScratchDataSizeInBytes: {prebuild_info.UpdateScratchDataSizeInBytes}")

        blas_result_buffer  = make_uav_buffer(self.device, prebuild_info.ResultDataMaxSizeInBytes, state=native.D3D12_RESOURCE_STATES.RAYTRACING_ACCELERATION_STRUCTURE)
        blas_scratch_buffer = make_uav_buffer(self.device, prebuild_info.ScratchDataSizeInBytes)

        instances = [
            D3D12_RAYTRACING_INSTANCE_DESC(
                Transform = (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                ),
                InstanceID = 0,
                InstanceMask = 0xFF,
                InstanceContributionToHitGroupIndex = 0,
                Flags = native.D3D12_RAYTRACING_INSTANCE_FLAGS.NONE,
                AccelerationStructure = blas_result_buffer.GetGPUVirtualAddress(),
            )
        ]

        instance_buffer = make_write_combined_buffer(self.device, size=len(instances) * ctypes.sizeof(D3D12_RAYTRACING_INSTANCE_DESC))
        if 1: # Upload
            mapped_ptr = instance_buffer.Map()
            array = (D3D12_RAYTRACING_INSTANCE_DESC * len(instances)).from_address(mapped_ptr)
            for i in range(len(instances)):
                array[i] = instances[i]
            instance_buffer.Unmap()

        tlas_inputs = native.D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS(
            Flags = native.D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS.PREFER_FAST_TRACE | native.D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS.ALLOW_UPDATE,
            NumDescs = 1,
            InstanceDescs = instance_buffer.GetGPUVirtualAddress()
        )
        tlas_prebuild_info = self.device.GetRaytracingAccelerationStructurePrebuildInfo(tlas_inputs)
        print(f"ResultDataMaxSizeInBytes: {tlas_prebuild_info.ResultDataMaxSizeInBytes}")
        print(f"ScratchDataSizeInBytes: {tlas_prebuild_info.ScratchDataSizeInBytes}")
        print(f"UpdateScratchDataSizeInBytes: {tlas_prebuild_info.UpdateScratchDataSizeInBytes}")

        tlas_result_buffer  = make_uav_buffer(self.device, tlas_prebuild_info.ResultDataMaxSizeInBytes, state=native.D3D12_RESOURCE_STATES.RAYTRACING_ACCELERATION_STRUCTURE)

        tlas_build_info = native.D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC(
            Dest = tlas_result_buffer.GetGPUVirtualAddress(),
            Source = 0,
            Scratch = blas_scratch_buffer.GetGPUVirtualAddress(),
            Inputs = tlas_inputs,
        )

        blas_build_info = native.D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC(
            Dest = blas_result_buffer.GetGPUVirtualAddress(),
            Source = 0,
            Scratch = blas_scratch_buffer.GetGPUVirtualAddress(),
            Inputs = as_inputs,
        )

        cmd_list = self.device.CreateCommandList(NodeMask=0, Type=native.D3D12_COMMAND_LIST_TYPE.DIRECT, Allocator=cmd_alloc)

        cmd_list.BuildRaytracingAccelerationStructure(blas_build_info)
        cmd_list.ResourceBarrier([native.D3D12_RESOURCE_BARRIER(
            UAV = native.D3D12_RESOURCE_UAV_BARRIER(
                Resource = blas_scratch_buffer
            )
        )])
        cmd_list.BuildRaytracingAccelerationStructure(tlas_build_info)

        cmd_list.Close()

        e = native.Event()
        fence = self.device.CreateFence(0, native.D3D12_FENCE_FLAGS.NONE)
        fence.SetEventOnCompletion(1, e)
        cmd_queue.ExecuteCommandLists([cmd_list])
        cmd_queue.Signal(fence, 1)
        e.Wait()

        # !Descriptor heap
        cpu_handle, gpu_handle = self.cbv_srv_uav_heap.get_next_descriptor_handle(16)
        srv_cpu_handle, srv_gpu_handle = self.cbv_srv_uav_heap.get_next_descriptor_handle(16)
        self.descriptor_table = gpu_handle
        self.srv_descriptor_table = srv_gpu_handle

        pitch               = 4 * 4 * width
        storage_buffer_size = pitch * height
        params_storage_size = 16 * (1 << 20); # 16mb should be enough for now

        params_buffer   = make_uav_buffer(self.device, params_storage_size)
        grads_buffer    = make_uav_buffer(self.device, params_storage_size)

        self.params_buffer = params_buffer
        self.grads_buffer  = grads_buffer

        # !UAV
        self.uav_texture    = make_uav_texture_2d(self.device, width=width, height=height, format=native.DXGI_FORMAT.R16G16B16A16_FLOAT)
        self.tmp_uav        = make_uav_texture_2d(self.device, width=width, height=height, format=native.DXGI_FORMAT.R16G16B16A16_FLOAT)
        self.viewz          = make_uav_texture_2d(self.device, width=width, height=height, format=native.DXGI_FORMAT.R32_FLOAT)
        self.normals         = make_uav_texture_2d(self.device, width=width, height=height, format=native.DXGI_FORMAT.R16G16B16A16_FLOAT)
        

        self.device.CreateUnorderedAccessView(
            Resource = self.uav_texture,
            Desc = native.D3D12_UNORDERED_ACCESS_VIEW_DESC(
                Format = native.DXGI_FORMAT.R16G16B16A16_FLOAT,
                Texture2D = native.D3D12_TEX2D_UAV(
                    MipSlice = 0,
                    PlaneSlice = 0
                )
            ),
            DestDescriptor = native.D3D12_CPU_DESCRIPTOR_HANDLE(cpu_handle.ptr + 0 * self.cbv_srv_uav_heap.descriptor_size),
            CounterResource = None
        )
        self.device.CreateUnorderedAccessView(
            Resource = self.tmp_uav,
            Desc = native.D3D12_UNORDERED_ACCESS_VIEW_DESC(
                Format = native.DXGI_FORMAT.R16G16B16A16_FLOAT,
                Texture2D = native.D3D12_TEX2D_UAV(
                    MipSlice = 0,
                    PlaneSlice = 0
                )
            ),
            DestDescriptor = native.D3D12_CPU_DESCRIPTOR_HANDLE(cpu_handle.ptr + 1 * self.cbv_srv_uav_heap.descriptor_size),
            CounterResource = None
        )
        self.device.CreateUnorderedAccessView(
                Resource=params_buffer,
                Desc = native.D3D12_UNORDERED_ACCESS_VIEW_DESC(
                    Format = native.DXGI_FORMAT.UNKNOWN,
                    Buffer = native.D3D12_BUFFER_UAV(
                        FirstElement = 0,
                        NumElements = params_storage_size // 4,
                        StructureByteStride = 4,
                        CounterOffsetInBytes = 0,
                        Flags = native.D3D12_BUFFER_UAV_FLAGS.NONE
                    )
                ),
                DestDescriptor = native.D3D12_CPU_DESCRIPTOR_HANDLE(cpu_handle.ptr + 2 * self.cbv_srv_uav_heap.descriptor_size),
                CounterResource = None
            )

        self.device.CreateUnorderedAccessView(
                Resource=grads_buffer,
                Desc = native.D3D12_UNORDERED_ACCESS_VIEW_DESC(
                    Format = native.DXGI_FORMAT.UNKNOWN,
                    Buffer = native.D3D12_BUFFER_UAV(
                        FirstElement = 0,
                        NumElements = params_storage_size // 4,
                        StructureByteStride = 4,
                        CounterOffsetInBytes = 0,
                        Flags = native.D3D12_BUFFER_UAV_FLAGS.NONE
                    )
                ),
                DestDescriptor = native.D3D12_CPU_DESCRIPTOR_HANDLE(cpu_handle.ptr + 3 * self.cbv_srv_uav_heap.descriptor_size),
                CounterResource = None
            )
        self.device.CreateUnorderedAccessView(
            Resource = self.viewz,
            Desc = native.D3D12_UNORDERED_ACCESS_VIEW_DESC(
                Format = native.DXGI_FORMAT.R32_FLOAT,
                Texture2D = native.D3D12_TEX2D_UAV(
                    MipSlice = 0,
                    PlaneSlice = 0
                )
            ),
            DestDescriptor = native.D3D12_CPU_DESCRIPTOR_HANDLE(cpu_handle.ptr + 4 * self.cbv_srv_uav_heap.descriptor_size),
            CounterResource = None
        )
        self.device.CreateUnorderedAccessView(
            Resource = self.normals,
            Desc = native.D3D12_UNORDERED_ACCESS_VIEW_DESC(
                Format = native.DXGI_FORMAT.R16G16B16A16_FLOAT,
                Texture2D = native.D3D12_TEX2D_UAV(
                    MipSlice = 0,
                    PlaneSlice = 0
                )
            ),
            DestDescriptor = native.D3D12_CPU_DESCRIPTOR_HANDLE(cpu_handle.ptr + 5 * self.cbv_srv_uav_heap.descriptor_size),
            CounterResource = None
        )

        # !SRV
        self.device.CreateShaderResourceView(
            Resource = None,
            Desc = native.D3D12_SHADER_RESOURCE_VIEW_DESC(
               RaytracingAccelerationStructure = native.D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV(
                   Location = tlas_result_buffer.GetGPUVirtualAddress()
               )
            ),
            DestDescriptor = native.D3D12_CPU_DESCRIPTOR_HANDLE(srv_cpu_handle.ptr + 0 * self.cbv_srv_uav_heap.descriptor_size)
        )

        self.device.CreateShaderResourceView(
            Resource = self.uav_texture,
            Desc = native.D3D12_SHADER_RESOURCE_VIEW_DESC(
                Format = native.DXGI_FORMAT.R16G16B16A16_FLOAT,
                Shader4ComponentMapping = native.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Texture2D = native.D3D12_TEX2D_SRV(
                    MipLevels = 1,
                    MostDetailedMip = 0,
                    PlaneSlice = 0,
                    ResourceMinLODClamp = 0.0
                )
            ),
            DestDescriptor = native.D3D12_CPU_DESCRIPTOR_HANDLE(srv_cpu_handle.ptr + 1 * self.cbv_srv_uav_heap.descriptor_size)
        )
        self.srv_texture_srv_gpu_descritpor = gpu_handle

        self.device.CreateShaderResourceView(
            Resource = mega_buffer,
            Desc = native.D3D12_SHADER_RESOURCE_VIEW_DESC(
                Format = native.DXGI_FORMAT.R32_TYPELESS,
                Buffer = native.D3D12_BUFFER_SRV(
                    FirstElement = 0,
                    NumElements = mega_buffer.GetDesc().Width // 4,
                    StructureByteStride = 0,
                    Flags = native.D3D12_BUFFER_SRV_FLAGS.RAW
                )
            ),
            DestDescriptor = native.D3D12_CPU_DESCRIPTOR_HANDLE(srv_cpu_handle.ptr + 2 * self.cbv_srv_uav_heap.descriptor_size)
        )

        self.device.CreateShaderResourceView(
            Resource = self.shader_geometry_descs_buffer,
            Desc = native.D3D12_SHADER_RESOURCE_VIEW_DESC(
                Format = native.DXGI_FORMAT.R32_TYPELESS,
                Buffer = native.D3D12_BUFFER_SRV(
                    FirstElement = 0,
                    NumElements = self.shader_geometry_descs_buffer.GetDesc().Width // 4,
                    StructureByteStride = 0,
                    Flags = native.D3D12_BUFFER_SRV_FLAGS.RAW
                )
            ),
            DestDescriptor = native.D3D12_CPU_DESCRIPTOR_HANDLE(srv_cpu_handle.ptr + 3 * self.cbv_srv_uav_heap.descriptor_size)
        )
        self.device.CreateShaderResourceView(
            Resource = self.sobol_buffer,
            Desc = native.D3D12_SHADER_RESOURCE_VIEW_DESC(
                Format = native.DXGI_FORMAT.R32_TYPELESS,
                Buffer = native.D3D12_BUFFER_SRV(
                    FirstElement = 0,
                    NumElements = self.sobol_buffer.GetDesc().Width // 4,
                    StructureByteStride = 0,
                    Flags = native.D3D12_BUFFER_SRV_FLAGS.RAW
                )
            ),
            DestDescriptor = native.D3D12_CPU_DESCRIPTOR_HANDLE(srv_cpu_handle.ptr + 4 * self.cbv_srv_uav_heap.descriptor_size)
        )

        self.device.CreateShaderResourceView(
            Resource = self.scrambling_tile_buffer,
            Desc = native.D3D12_SHADER_RESOURCE_VIEW_DESC(
                Format = native.DXGI_FORMAT.R32_TYPELESS,
                Buffer = native.D3D12_BUFFER_SRV(
                    FirstElement = 0,
                    NumElements = self.scrambling_tile_buffer.GetDesc().Width // 4,
                    StructureByteStride = 0,
                    Flags = native.D3D12_BUFFER_SRV_FLAGS.RAW
                )
            ),
            DestDescriptor = native.D3D12_CPU_DESCRIPTOR_HANDLE(srv_cpu_handle.ptr + 5 * self.cbv_srv_uav_heap.descriptor_size)
        )

        self.device.CreateShaderResourceView(
            Resource = self.ranking_tile_buffer,
            Desc = native.D3D12_SHADER_RESOURCE_VIEW_DESC(
                Format = native.DXGI_FORMAT.R32_TYPELESS,
                Buffer = native.D3D12_BUFFER_SRV(
                    FirstElement = 0,
                    NumElements = self.ranking_tile_buffer.GetDesc().Width // 4,
                    StructureByteStride = 0,
                    Flags = native.D3D12_BUFFER_SRV_FLAGS.RAW
                )
            ),
            DestDescriptor = native.D3D12_CPU_DESCRIPTOR_HANDLE(srv_cpu_handle.ptr + 6 * self.cbv_srv_uav_heap.descriptor_size)
        )


        # keep references
        self.mega_buffer        = mega_buffer
        self.tlas_result_buffer = tlas_result_buffer
        self.blas_result_buffer = blas_result_buffer

        cmd_alloc.Reset()

        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.on_frame)
        self.render_timer.start(0)

        pass

    def on_frame(self):

        self.cur_time = time.time()
        dt = self.cur_time - self.last_time
        self.last_time = self.cur_time
        # print(f"dt = {dt}")

        if self.child.mouse_in_window == False:
            time.sleep(0.1)
            return

        # !Imgui
        if ctypes.windll.user32.IsWindow(self.hwnd) == 0:
            print_red("Window is closed")
            self.render_timer.stop()
            on_close()
            return
        try:
            window_width, window_height = native.GetWindowSize(self.hwnd)
        except Exception as e:
            print_red(f"Failed to get window size: {e}")
            self.render_timer.stop()
            on_close()
            return
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
        
        cmd_list.SetPipelineState(self.rt_pso)
        cmd_list.SetComputeRootSignature(self.rt_signature)
        cmd_list.SetDescriptorHeaps([self.cbv_srv_uav_heap.heap])
        cmd_list.SetComputeRootDescriptorTable(RootParameterIndex  = 0, BaseDescriptor = self.descriptor_table)
        cmd_list.SetComputeRootDescriptorTable(RootParameterIndex  = 1, BaseDescriptor = self.srv_descriptor_table)
        cmd_list.SetComputeRootConstantBufferView(
            RootParameterIndex  = 2,
            BufferLocation      = self.cbuffer_wb.GetGPUVirtualAddress() + back_buffer_idx * ctypes.sizeof(CBuffer)
        )

        camera_speed = 0.1
        if key_press_map.get(Qt.Key_W, False): camera.move_forward(1.0 * camera_speed)
        if key_press_map.get(Qt.Key_S, False): camera.move_forward(-1.0 * camera_speed)
        if key_press_map.get(Qt.Key_A, False): camera.move_right(-1.0 * camera_speed)
        if key_press_map.get(Qt.Key_D, False): camera.move_right(1.0 * camera_speed)
        if key_press_map.get(Qt.Key_Q, False): camera.move_up(-1.0 * camera_speed)
        if key_press_map.get(Qt.Key_E, False): camera.move_up(1.0 * camera_speed)

        # print(f"camera.pos = {camera.pos}")

        camera.update()
        self.cbuffer_wb_arr[back_buffer_idx].frustum_x      = (camera.frustum_x.x, camera.frustum_x.y, camera.frustum_x.z)
        self.cbuffer_wb_arr[back_buffer_idx].frustum_y      = (camera.frustum_y.x, camera.frustum_y.y, camera.frustum_y.z)
        self.cbuffer_wb_arr[back_buffer_idx].frustum_z      = (camera.frustum_z.x, camera.frustum_z.y, camera.frustum_z.z)
        self.cbuffer_wb_arr[back_buffer_idx].half_fov_tan   = camera.half_fov_tan
        self.cbuffer_wb_arr[back_buffer_idx].aspect         = camera.aspect
        self.cbuffer_wb_arr[back_buffer_idx].camera_pos     = (camera.pos.x, camera.pos.y, camera.pos.z)
        self.cbuffer_wb_arr[back_buffer_idx].frame_idx      = self.frame_idx

        cmd_list.Dispatch(self.uav_texture.GetDesc().Width // 8, self.uav_texture.GetDesc().Height, 1)

        if self.frame_idx == 0:
            cmd_list.SetPipelineState(self.Initialize_pso)
            cmd_list.Dispatch(self.uav_texture.GetDesc().Width // 8, self.uav_texture.GetDesc().Height // 8, 1)
    
        cmd_list.ResourceBarrier([native.D3D12_RESOURCE_BARRIER(
            UAV = native.D3D12_RESOURCE_UAV_BARRIER(
                Resource = self.grads_buffer
            )
        )])

        cmd_list.SetPipelineState(self.Main_pso)
        cmd_list.Dispatch(128, 1, 1)
        
        cmd_list.ResourceBarrier([native.D3D12_RESOURCE_BARRIER(
            UAV = native.D3D12_RESOURCE_UAV_BARRIER(
                Resource = self.grads_buffer
            )
        )])
        cmd_list.SetPipelineState(self.Backwards_pso)
        cmd_list.Dispatch(self.uav_texture.GetDesc().Width // 8, self.uav_texture.GetDesc().Height // 8, 1)

        cmd_list.ResourceBarrier([native.D3D12_RESOURCE_BARRIER(
            UAV = native.D3D12_RESOURCE_UAV_BARRIER(
                Resource = self.grads_buffer
            )
        )])
        cmd_list.SetPipelineState(self.Inference_pso)
        cmd_list.Dispatch(self.uav_texture.GetDesc().Width // 8, self.uav_texture.GetDesc().Height // 8, 1)

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
            BaseDescriptor = self.srv_texture_srv_gpu_descritpor
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

on_close()
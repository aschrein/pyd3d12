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


import enum
from dataclasses import dataclass
import random
import os, sys
import ctypes
import argparse
from py.utils import *
import numpy as np
import torch
from py.torch_utils import *
from py.dxc import *
from piq import LPIPS

NUM_REGISTERS = 64 # each register is f32

class Buffer:
    def __init__(self, max_size_bytes: int):
        self.buffer = np.zeros((max_size_bytes,), dtype=np.uint8)
        self.size_bytes = max_size_bytes
        self.cursor = 0

    def write_u32(self, value: int):
        assert self.cursor + 4 <= self.size_bytes
        self.buffer[self.cursor:self.cursor+4] = np.frombuffer(ctypes.c_uint32(value), dtype=np.uint8)
        self.cursor += 4
    
    def write_f32(self, value: float):
        assert self.cursor + 4 <= self.size_bytes
        self.buffer[self.cursor:self.cursor+4] = np.frombuffer(ctypes.c_float(value), dtype=np.uint8)
        self.cursor += 4
    
    def reset(self):
        self.cursor = 0

    def get_ptr(self):
        return self.buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

@dataclass
class Op:
    pass

def get_opcode(cls):
    return hash(cls) & 0xFFFF

class BinOps(enum.Enum):
    ADD = 0
    SUB = 1
    MUL = 2
    MAX = 3
    MIN = 4
    STEP = 5
    # DIV = 3

@dataclass
class BinOp(Op):
    op  : BinOps
    dest: int
    src1: int
    src2: int

    def serialize(self, buffer: Buffer):
        buffer.write_u32(get_opcode(BinOp))
        buffer.write_u32(self.op.value)
        buffer.write_u32(self.dest)
        buffer.write_u32(self.src1)
        buffer.write_u32(self.src2)

    @staticmethod
    def sample():
        rnd_binop = random.choice(list(BinOps))
        return BinOp(op=rnd_binop, dest=random.randint(0, NUM_REGISTERS-1), src1=random.randint(0, NUM_REGISTERS-1), src2=random.randint(0, NUM_REGISTERS-1))

class UnOps(enum.Enum):
    NEG = 0
    NOP = 1
    SIN = 2
    COS = 3
    EXP = 4
    LOG = 5
    SQUARE = 6
    SQRT = 7

@dataclass
class UnOp(Op):
    name: UnOps
    dest: int
    src: int

    def serialize(self, buffer: Buffer):
        buffer.write_u32(get_opcode(UnOp))
        buffer.write_u32(self.name.value)
        buffer.write_u32(self.dest)
        buffer.write_u32(self.src)

    @staticmethod
    def sample():
        rnd_unop = random.choice(list(UnOps))
        return UnOp(name=rnd_unop, dest=random.randint(0, NUM_REGISTERS-1), src=random.randint(0, NUM_REGISTERS-1))

@dataclass
class MovConstOp(Op):
    dest: int
    value: float

    def serialize(self, buffer: Buffer):
        buffer.write_u32(get_opcode(MovConstOp))
        buffer.write_u32(self.dest)
        buffer.write_f32(self.value)

    @staticmethod
    def sample():
        return MovConstOp(dest=random.randint(0, NUM_REGISTERS-1), value=random.gauss(0, 1))

# random.seed(42)

OP_LIST = [
    BinOp,
    UnOp,
    MovConstOp,
]
OP_INDICES = {
    BinOp:      0,
    UnOp:       1,
    MovConstOp: 2,
}

def sample_op():
    op_type = random.choice(OP_LIST)
    return op_type.sample()


if 0:
    print("Sampled operations:")
    for _ in range(16):
        op = sample_op()
        print(f" - {op}")

    add = BinOp(op=BinOps.ADD, dest=0, src1=1, src2=2)
    # print(f"ADD opcode: {add.opcode:#06x}")

    # load_const r0, f32x4(0.0, 1.0, 2.0, 3.0)
    # load_uv r1
    # mov r2, r0
    # neg r3, r1
    # add r4, r2, r3

args = argparse.ArgumentParser()
args.add_argument("--build", type=str, default="Release")
args.add_argument("--wait_for_debugger_present", action="store_true")
args.add_argument("--load_rdoc", action="store_true")
args.add_argument("--load_pix", action="store_true")
args.add_argument("--make_window", action="store_true")
args = args.parse_args()
set_build_type(args.build)

from py.dxc import *
from py.rdoc import *
from py.d3d12 import *
from py.pix import *

IMAGE_SIZE = 64

if 0:
    hlsl_header = ""

    hlsl_header += """
    enum Opcode {
    """
    for op_cls in OP_LIST:
        hlsl_header += f"    {op_cls.__name__.upper()} = {OP_INDICES[op_cls]},\n"
    hlsl_header += """
    };
    """

    hlsl_header += """
    enum BinOps {
    """
    for binop in BinOps:
        hlsl_header += f"    {binop.name} = {binop.value},\n" 
    hlsl_header += """
    };
    """

    hlsl_header += """
    enum UnOps {
    """
    for unop in UnOps:
        hlsl_header += f"    {unop.name} = {unop.value},\n"
    hlsl_header += """
    };
    """

    # print("Generated HLSL header:")
    # print(hlsl_header)

    native = find_native_module("native")

    debug = native.ID3D12Debug()
    debug.EnableDebugLayer()

    factory = native.IDXGIFactory()
    adapters = factory.EnumAdapters()

    for adapter in adapters:
        print(f"Adapter: {adapter.GetDesc().Description}")
        break

    device = native.CreateDevice(adapters[0], native.D3D_FEATURE_LEVEL._11_0)

    dxc_ctx = DXCContext()

    src = f"""
    //js

    #define u32 uint
    #define i32 int
    #define u32x2 uint2
    #define i32x2 int2
    #define f32 float
    #define f32x2 float2
    #define f32x3 float3
    #define f32x4 float4

    #define ROOT_SIGNATURE_MACRO                                                                    \
        "DescriptorTable("                                                                          \
        "UAV("                                                                                       \
            "u0, NumDescriptors = 16, space=0, flags = DESCRIPTORS_VOLATILE, offset=0) "            \
        "), "                                                                                       \
        "DescriptorTable("                                                                          \
        "SRV("                                                                                       \
            "t0, NumDescriptors = 16, space=0, flags = DESCRIPTORS_VOLATILE, offset=0) "             \
        "), "                                                                                       \
        "RootConstants(b0, num32BitConstants=32, space = 0), "                                      \

    struct RootConstants {{
        u32 start_op_offset;
        u32 num_ops;
        u32 result_offset;
    }};

    ConstantBuffer<RootConstants> rc : register(b0, space0);

    RWByteAddressBuffer g_rw_memory : register(u0, space0);
    RWByteAddressBuffer g_rw_result : register(u1, space0);

    #define FETCH_NEXT_DWORD(type) g_rw_memory.Load<type>(memory_offset); memory_offset += 4;

    {hlsl_header}

    #define SUBGROUP_SIZE 32
    #define NUM_REGISTERS 16
    groupshared f32 registers[SUBGROUP_SIZE][NUM_REGISTERS];

    [RootSignature(ROOT_SIGNATURE_MACRO)] //
    [numthreads(32, 1, 1)]                //
    void Main(i32x2 global_idx : SV_DispatchThreadID) {{

        const u32 lane_idx = WaveGetLaneIndex();

        const u32 img_size = {IMAGE_SIZE};
        const f32x2 uv = f32x2(
            (f32(global_idx.x) + f32(0.5)) / f32(img_size),
            (f32(global_idx.y) + f32(0.5)) / f32(img_size)
        );

        u32 memory_offset = rc.start_op_offset;
        const u32 thread_dst_offset_bytes = rc.result_offset + global_idx * 3 * 4;

        for (i32 r = 0; r < NUM_REGISTERS; r++) {{
            registers[lane_idx][r] = f32(0.0);
        }}
        registers[lane_idx][0] = f32(uv.x);
        registers[lane_idx][1] = f32(uv.y);

        for (int op_idx = 0; op_idx < rc.num_ops; op_idx++) {{
            // Process each operation
            const u32 opcode = FETCH_NEXT_DWORD(u32);

            GroupMemoryBarrierWithGroupSync(); // should be a no op here since all lanes execute in lockstep

            switch (opcode) {{
                case Opcode::BINOP: {{
                    const u32 binop_type    = FETCH_NEXT_DWORD(u32);
                    const u32 dest_idx      = FETCH_NEXT_DWORD(u32);
                    const u32 src1_idx      = FETCH_NEXT_DWORD(u32);
                    const u32 src2_idx      = FETCH_NEXT_DWORD(u32);
                    const f32 src1          = registers[lane_idx][src1_idx];
                    const f32 src2          = registers[lane_idx][src2_idx];

                    f32 result = 0.0;

                    switch (binop_type) {{
                        case BinOps::ADD: result = src1 + src2; break;
                        case BinOps::SUB: result = src1 - src2; break;
                        case BinOps::MUL: result = src1 * src2; break;
                        case BinOps::DIV: result = src1 / src2; break;
                        case BinOps::MAX: result = max(src1, src2); break;
                        case BinOps::MIN: result = min(src1, src2); break;
                        default: break;
                    }}

                    registers[lane_idx][dest_idx] = result;

                }} break;

                case Opcode::UNOP: {{
                    const u32 unop_type     = FETCH_NEXT_DWORD(u32);
                    const u32 dest_idx      = FETCH_NEXT_DWORD(u32);
                    const u32 src_idx       = FETCH_NEXT_DWORD(u32);

                    const f32 src = registers[lane_idx][src_idx];
                    f32 result = 0.0;

                    switch (unop_type) {{
                        case UnOps::NEG: result = -src; break;
                        case UnOps::SIN: result = sin(src); break;
                        case UnOps::COS: result = cos(src); break;

                        default: break;
                    }}

                    registers[lane_idx][dest_idx] = result;

                }} break;

                case Opcode::MovConstOp: {{
                    const u32 dest_idx      = FETCH_NEXT_DWORD(u32);
                    const f32 value         = FETCH_NEXT_DWORD(f32);

                    registers[lane_idx][dest_idx] = value;

                }} break;

                default: {{
                    // Unknown opcode
                }} break;
            }}

    }}

    const f32 result_0 = registers[lane_idx][0];
    const f32 result_1 = registers[lane_idx][1];
    const f32 result_2 = registers[lane_idx][2];

    g_rw_result.Store<f32>(thread_dst_offset_bytes + 0 * 4, result_0);
    g_rw_result.Store<f32>(thread_dst_offset_bytes + 1 * 4, result_1);
    g_rw_result.Store<f32>(thread_dst_offset_bytes + 2 * 4, result_2);

    }}

    ;//js

    """

    with open(get_or_create_tmp_folder() / "vm_cs.hlsl", "w") as f:
        f.write(src)

    bytecode = dxc_ctx.compile_to_dxil(source = src, args = "-E Main -T cs_6_5")

    signature = device.CreateRootSignature(
        Bytes = bytecode
    )
    assert signature is not None
    Main_pso = device.CreateComputePipelineState(native.D3D12_COMPUTE_PIPELINE_STATE_DESC(
        RootSignature = signature,
        CS = native.D3D12_SHADER_BYTECODE(bytecode),
        NodeMask = 0,
        CachedPSO = None,
        Flags = native.D3D12_PIPELINE_STATE_FLAGS.NONE
    ))
    assert Main_pso is not None


    memory = Buffer(128 * (1 << 20))  # 128 MB buffer
    gpu_buffer = make_write_combined_buffer(device, memory.size_bytes)
    READBACK_SIZE = 128 * (1 << 20)
    result_buffer = make_readback_buffer(device, READBACK_SIZE)

    def copy_to_gpu():
        dst_data_ptr = gpu_buffer.Map(0, None)
        ctypes.memmove(dst_data_ptr, memory.get_ptr(), memory.size_bytes)
        gpu_buffer.Unmap(0, None)

    def read_from_gpu():
        src_data_ptr = result_buffer.Map(0, None)
        new_buffer = np.zeros((READBACK_SIZE//4,), dtype=np.float32)
        ctypes.memmove(new_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), src_data_ptr, READBACK_SIZE)
        result_buffer.Unmap(0, None)
        return new_buffer

    copy_to_gpu()

# Training
torch_device = torch.device("cuda")
x_coords     = torch.linspace(0.5, IMAGE_SIZE - 0.5, IMAGE_SIZE, device=torch_device).unsqueeze(0).repeat(IMAGE_SIZE, 1) / IMAGE_SIZE
y_coords     = torch.linspace(0.5, IMAGE_SIZE - 0.5, IMAGE_SIZE, device=torch_device).unsqueeze(1).repeat(1, IMAGE_SIZE) / IMAGE_SIZE
uv           = torch.stack([x_coords, y_coords], dim=0)  # 2 x H x W

class LearnableOp(torch.nn.Module):
    def __init__(self, num_registers: int, device: torch.device):
        super().__init__()
        self.dest_selector = torch.nn.Parameter(torch.randn((num_registers, 1, 1), dtype=torch.float32, device=device))
        self.src0_selector = torch.nn.Parameter(torch.randn((num_registers, 1, 1), dtype=torch.float32, device=device))
        self.src1_selector = torch.nn.Parameter(torch.randn((num_registers, 1, 1), dtype=torch.float32, device=device))
        self.value         = torch.nn.Parameter(torch.randn((1, 1, 1), dtype=torch.float32, device=device))
        self.op_select     = torch.nn.Parameter(torch.randn((len(OP_LIST), 1, 1), dtype=torch.float32, device=device))
        self.binop_select  = torch.nn.Parameter(torch.randn((len(BinOps), 1, 1), dtype=torch.float32, device=device))
        self.unop_select   = torch.nn.Parameter(torch.randn((len(UnOps), 1, 1), dtype=torch.float32, device=device))
        self.frozen_selectors = False

    def freeze_selectors(self, freeze=True):
        if self.frozen_selectors == freeze:
            return
        self.frozen_selectors = freeze

        self.dest_selector.requires_grad = not freeze
        self.src0_selector.requires_grad = not freeze
        self.src1_selector.requires_grad = not freeze
        self.op_select.requires_grad     = not freeze
        self.binop_select.requires_grad  = not freeze
        self.unop_select.requires_grad   = not freeze

        dest_mask   = torch.nn.functional.softmax(self.dest_selector, dim=0)
        src0_mask   = torch.nn.functional.softmax(self.src0_selector, dim=0)
        src1_mask   = torch.nn.functional.softmax(self.src1_selector, dim=0)
        binop_mask  = torch.nn.functional.softmax(self.binop_select, dim=0)
        unop_mask   = torch.nn.functional.softmax(self.unop_select, dim=0)
        op_mask     = torch.nn.functional.softmax(self.op_select, dim=0)

        dest_mask_collapsed_idx   = dest_mask.argmax(dim=0)
        src0_mask_collapsed_idx   = src0_mask.argmax(dim=0)
        src1_mask_collapsed_idx   = src1_mask.argmax(dim=0)
        binop_mask_collapsed_idx  = binop_mask.argmax(dim=0)
        unop_mask_collapsed_idx   = unop_mask.argmax(dim=0)
        op_mask_collapsed_idx     = op_mask.argmax(dim=0)

        dest_mask_one_hot   = torch.nn.functional.one_hot(dest_mask_collapsed_idx,  num_classes=NUM_REGISTERS).float().permute(2, 0, 1)
        src0_mask_one_hot   = torch.nn.functional.one_hot(src0_mask_collapsed_idx,  num_classes=NUM_REGISTERS).float().permute(2, 0, 1)
        src1_mask_one_hot   = torch.nn.functional.one_hot(src1_mask_collapsed_idx,  num_classes=NUM_REGISTERS).float().permute(2, 0, 1)
        binop_mask_one_hot  = torch.nn.functional.one_hot(binop_mask_collapsed_idx, num_classes=len(BinOps)).float().permute(2, 0, 1)
        unop_mask_one_hot   = torch.nn.functional.one_hot(unop_mask_collapsed_idx,  num_classes=len(UnOps)).float().permute(2, 0, 1)
        op_mask_one_hot     = torch.nn.functional.one_hot(op_mask_collapsed_idx,    num_classes=len(OP_LIST)).float().permute(2, 0, 1)
        
        # dest_mask   = dest_mask  + (dest_mask_one_hot - dest_mask).detach()
        # src0_mask   = src0_mask  + (src0_mask_one_hot - src0_mask).detach()
        # src1_mask   = src1_mask  + (src1_mask_one_hot - src1_mask).detach()
        # op_mask     = op_mask    + (op_mask_one_hot - op_mask).detach()
        # binop_mask  = binop_mask + (binop_mask_one_hot - binop_mask).detach()
        # unop_mask   = unop_mask  + (unop_mask_one_hot - unop_mask).detach()

        self.dest_selector.data = torch.lerp(self.dest_selector.data, dest_mask_one_hot.data, 0.5)
        self.src0_selector.data = torch.lerp(self.src0_selector.data, src0_mask_one_hot.data, 0.5)
        self.src1_selector.data = torch.lerp(self.src1_selector.data, src1_mask_one_hot.data, 0.5)
        self.op_select.data     = torch.lerp(self.op_select.data,     op_mask_one_hot.data, 0.5)
        self.binop_select.data  = torch.lerp(self.binop_select.data,  binop_mask_one_hot.data, 0.5)
        self.unop_select.data   = torch.lerp(self.unop_select.data,   unop_mask_one_hot.data, 0.5)

        # self.dest_selector.data = self.dest_selector / self.dest_selector.sum(dim=0, keepdim=True)
        # self.src0_selector.data = self.src0_selector / self.src0_selector.sum(dim=0, keepdim=True)
        # self.src1_selector.data = self.src1_selector / self.src1_selector.sum(dim=0, keepdim=True)
        # self.op_select.data     = self.op_select / self.op_select.sum(dim=0, keepdim=True)
        # self.binop_select.data  = self.binop_select / self.binop_select.sum(dim=0, keepdim=True)
        # self.unop_select.data   = self.unop_select / self.unop_select.sum(dim=0, keepdim=True)

    def unfreeze_selectors(self):
        self.freeze_selectors(freeze=False)

    def clone(self):
        new_op = LearnableOp(NUM_REGISTERS, torch_device)
        for param, new_param in zip(self.parameters(), new_op.parameters()):
            new_param.data = param.data.clone()
        return new_op

    def forward(self, registers, collapse=False):
        dest_mask   = torch.nn.functional.softmax(1.0 * self.dest_selector, dim=0)
        src0_mask   = torch.nn.functional.softmax(1.0 * self.src0_selector, dim=0)
        src1_mask   = torch.nn.functional.softmax(1.0 * self.src1_selector, dim=0)
        binop_mask  = torch.nn.functional.softmax(1.0 * self.binop_select, dim=0)
        unop_mask   = torch.nn.functional.softmax(1.0 * self.unop_select, dim=0)
        op_mask     = torch.nn.functional.softmax(1.0 * self.op_select, dim=0)

        if collapse:
            # dest_mask   = torch.nn.functional.softmax(8.0 * self.dest_selector, dim=0)
            # src0_mask   = torch.nn.functional.softmax(8.0 * self.src0_selector, dim=0)
            # src1_mask   = torch.nn.functional.softmax(8.0 * self.src1_selector, dim=0)
            # binop_mask  = torch.nn.functional.softmax(8.0 * self.binop_select, dim=0)
            # unop_mask   = torch.nn.functional.softmax(8.0 * self.unop_select, dim=0)
            # op_mask     = torch.nn.functional.softmax(8.0 * self.op_select, dim=0)

            if 1:
                dest_mask_collapsed_idx   = dest_mask.argmax(dim=0)
                src0_mask_collapsed_idx   = src0_mask.argmax(dim=0)
                src1_mask_collapsed_idx   = src1_mask.argmax(dim=0)
                binop_mask_collapsed_idx  = binop_mask.argmax(dim=0)
                unop_mask_collapsed_idx   = unop_mask.argmax(dim=0)
                op_mask_collapsed_idx     = op_mask.argmax(dim=0)
                dest_mask_one_hot   = torch.nn.functional.one_hot(dest_mask_collapsed_idx,  num_classes=NUM_REGISTERS).float().permute(2, 0, 1)
                src0_mask_one_hot   = torch.nn.functional.one_hot(src0_mask_collapsed_idx,  num_classes=NUM_REGISTERS).float().permute(2, 0, 1)
                src1_mask_one_hot   = torch.nn.functional.one_hot(src1_mask_collapsed_idx,  num_classes=NUM_REGISTERS).float().permute(2, 0, 1)
                binop_mask_one_hot  = torch.nn.functional.one_hot(binop_mask_collapsed_idx, num_classes=len(BinOps)).float().permute(2, 0, 1)
                unop_mask_one_hot   = torch.nn.functional.one_hot(unop_mask_collapsed_idx,  num_classes=len(UnOps)).float().permute(2, 0, 1)
                op_mask_one_hot     = torch.nn.functional.one_hot(op_mask_collapsed_idx,    num_classes=len(OP_LIST)).float().permute(2, 0, 1)
                dest_mask   = dest_mask_one_hot # dest_mask   + (dest_mask_one_hot - dest_mask).detach()
                src0_mask   = src0_mask_one_hot # src0_mask   + (src0_mask_one_hot - src0_mask).detach()
                src1_mask   = src1_mask_one_hot # src1_mask   + (src1_mask_one_hot - src1_mask).detach()
                binop_mask  = binop_mask_one_hot # binop_mask  + (binop_mask_one_hot - binop_mask).detach()
                unop_mask   = unop_mask_one_hot # unop_mask   + (unop_mask_one_hot - unop_mask).detach()
                op_mask     = op_mask_one_hot # op_mask     + (op_mask_one_hot - op_mask).detach()

                # assert (dest_mask.isnan().any() == False)
                # assert (src0_mask.isnan().any() == False)
                # assert (src1_mask.isnan().any() == False)
                # assert (binop_mask.isnan().any() == False)
                # assert (unop_mask.isnan().any() == False)
                # assert (op_mask.isnan().any() == False)

        src0 = (src0_mask * registers).sum(dim=0, keepdim=True)
        # src1 = (src1_mask * registers).sum(dim=0, keepdim=True)
        dst  = (dest_mask  * registers).sum(dim=0, keepdim=True)

        # BinOp
        binop_add_op = dst + src0
        binop_sub_op = dst - src0
        binop_mul_op = dst * src0
        binop_div_op = dst / (src0.sign() * (src0.abs().clamp(min=1.0e-3)))  # avoid div by zero

        softmax_src = torch.nn.functional.softmax(2.0 * torch.cat((dst, src0), dim=0), dim=0)

        binop_max_op = dst * softmax_src[0:1, :, :] + src0 * softmax_src[1:2, :, :]
        binop_min_op = dst * softmax_src[1:2, :, :] + src0 * softmax_src[0:1, :, :]

        binop_result = (
            binop_add_op * binop_mask[BinOps.ADD.value] +
            binop_sub_op * binop_mask[BinOps.SUB.value] +
            binop_mul_op * binop_mask[BinOps.MUL.value] +
            # binop_div_op * binop_mask[BinOps.DIV.value]
            binop_max_op * binop_mask[BinOps.MAX.value] +
            binop_min_op * binop_mask[BinOps.MIN.value] +
            (1.0 - softmax_src[0:1, :, :]) * binop_mask[BinOps.STEP.value]
        )

        # assert binop_result.isnan().any() == False

        # UnOp
        unop_neg_op    = -dst
        unop_sin_op    = torch.sin(dst)
        unop_cos_op    = torch.cos(dst)
        unop_exp_op    = torch.exp(dst.clamp(max=4.0))  # avoid overflow
        unop_log_op    = torch.log(dst.clamp(min=1.0e-3).abs())
        unop_sqrt_op   = torch.sqrt(dst.abs())
        unop_square_op = torch.square(dst)
        unop_nop    = src0 # copy src to dest
        unop_result =  (
            unop_neg_op * unop_mask[UnOps.NEG.value] +
            unop_sin_op * unop_mask[UnOps.SIN.value] +
            unop_cos_op * unop_mask[UnOps.COS.value] +
            unop_exp_op * unop_mask[UnOps.EXP.value] +
            unop_log_op * unop_mask[UnOps.LOG.value] +
            unop_nop     * unop_mask[UnOps.NOP.value] +
            unop_square_op * unop_mask[UnOps.SQUARE.value] +
            unop_sqrt_op * unop_mask[UnOps.SQRT.value]
        )
        # assert unop_result.isnan().any() == False

        store_const_op =  self.value

        result = registers * (1.0 - dest_mask) + (
            binop_result   * op_mask[OP_INDICES[BinOp]] +
            unop_result    * op_mask[OP_INDICES[UnOp]] +
            store_const_op * op_mask[OP_INDICES[MovConstOp]]
        ) * dest_mask

        masks = torch.cat([
            dest_mask,
            src0_mask,
            src1_mask,
            binop_mask,
            unop_mask,
            op_mask
        ], dim=0)

        return result, masks

import PIL.Image as Image
import torchvision.transforms as transforms
target_image = Image.open("data\\MNIST\\dataset-part1\\cat_10.png")
target_image = transforms.ToTensor()(target_image).unsqueeze(0).to(torch_device)  # 1 x 3 x H x W

num_epochs = 100000

src_registers    = torch.zeros((NUM_REGISTERS, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32, device=torch_device)
src_registers[0:1, :, :] = uv[0:1, :, :] * 2.0 - 1.0
src_registers[1:2, :, :] = uv[1:2, :, :] * 2.0 - 1.0

# import math
# src_registers[2:3, :, :] = torch.sin(src_registers[0:1, :, :] * math.pi * 1.0)
# src_registers[3:4, :, :] = torch.sin(src_registers[1:2, :, :] * math.pi * 1.0)
# src_registers[4:5, :, :] = torch.sin(src_registers[0:1, :, :] * math.pi * 2.0)
# src_registers[5:6, :, :] = torch.sin(src_registers[1:2, :, :] * math.pi * 2.0)


lpips = LPIPS().to(torch_device)

# Pre select

NUM_OPS = 16

ops = []

min_loss = 1.0e10

for NUM_OPS in range(1, 256):

    print(f"{CONSOLE_COLOR_BLUE} === Training with {NUM_OPS} operations === {CONSOLE_COLOR_RESET}")

    ops.append(LearnableOp(NUM_REGISTERS, torch_device))

    for j in range(10):

        best_ops = [op.clone() for op in ops]
        # min_loss = 1.0e10

        for i in range(0):
            # print(f"Preselection iteration {i:04d}")

            ops      = ops.copy()
            for param in [param for op in ops for param in op.parameters()]:
                l = random.random() * 0.1
                param.data = param.data * (1.0 - l) + torch.randn_like(param) * l

            # for op in ops:
            #     op.collapse()

            registers = src_registers.clone()
            for op in ops:
                registers, masks = op(registers)

            output_image = registers[-3:, :, :]
            loss = torch.nn.functional.mse_loss(output_image, target_image)

            # loss = loss + lpips(output_image.unsqueeze(0), target_image)

            if loss.isnan().any():
                print(f"Invalid loss detected")
                continue

            if loss.item() < min_loss:
                min_loss = loss.item()
                best_ops = ops
                print(f"Preselection Loss = {loss.item():.6f}")

        ops = [op.clone() for op in best_ops]

        optimizer = torch.optim.AdamW([param for op in ops for param in op.parameters()], lr=1e-2, weight_decay=1e-2)
        # optimizer = torch.optim.AdamW([param for param in ops[-1].parameters()], lr=1e-2, weight_decay=1e-2)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=110, power=1.0)

        # registers    = src_registers.clone()
        # for op in ops[:-1]:
        #     registers, masks = op(registers)
        #     # op.collapse()

        # last_registers = registers.detach().clone()


        for epoch in range(110):
            
            loss = 0.0

           # if epoch < 50:
           #     for op in ops:
           #         op.freeze_selectors()
           # else:
           #     for op in ops:
           #         op.unfreeze_selectors()

            # for op in ops:
            #     op.collapse()

            # registers = last_registers.clone()
            registers = src_registers.clone()

            for op in ops:
                registers, masks = op(registers)
                
                # op.collapse()
                    
                # registers, masks = ops[-1](registers)

                # loss = loss + 0.1 * -((masks + 1.0e-6).log() * masks).mean()  # encourage sharp selections

            output_image = registers[-3:, :, :]

            mse_loss = torch.nn.functional.mse_loss(output_image, target_image + torch.randn_like(target_image) * 0.05)
            
            # lpips_loss = lpips(output_image.unsqueeze(0), target_image)

            loss = loss + mse_loss # + lpips_loss

            if loss.isnan().any():
                print(f"Invalid loss detected, resetting to best ops")
                ops = [op.clone() for op in best_ops]
                break
        
            min_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            print(f"Epoch {epoch:04d}: Loss = {loss.item():.6f}")

            if epoch % 16 == 0:
                registers = src_registers.clone()
                for op in ops:
                    registers, masks = op(registers, collapse=False)
                    # registers, masks = op(registers)
                output_image = registers[-3:, :, :]

                dds = dds_from_tensor(output_image.unsqueeze(0))  # 1 x 3 x H x W
                dds.save(".tmp/output_image.dds")

                registers = src_registers.clone()
                for op in ops:
                    registers, masks = op(registers, collapse=True)
                    # registers, masks = op(registers)
                output_image = registers[-3:, :, :]

                dds = dds_from_tensor(output_image.unsqueeze(0))  # 1 x 3 x H x W
                dds.save(".tmp/output_image_collapsed.dds")

    # uv_dds       = dds_from_tensor(uv.unsqueeze(0))  # 1 x 2 x H x W
    # uv_dds.save(get_or_create_tmp_folder() / "uv.dds")


# MIT License
# Copyright (c) 2025 Anton Schreiner

import math
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
//js
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define f32     float
#define i32     int32_t
#define i8      int8_t
#define u32     uint32_t
#define u64     uint64_t
#define INT8_MAX  127

#define ifor(N) for(i32 i=0;i<(N);i++)
#define jfor(N) for(i32 j=0;j<(N);j++)
#define kfor(N) for(i32 k=0;k<(N);k++)
#define xfor(N) for(i32 x=0;x<(N);x++)
#define yfor(N) for(i32 y=0;y<(N);y++)

__device__ __forceinline__ i8 quantize(f32 val, f32 inv_scale) {
    return static_cast<i8>(min(max(rintf(val * inv_scale), -127.0f), 127.0f));
}

// Compute Q, K, V projections for a single head
// tokens:   [num_tokens, 16]   - input embeddings (int8)
// Q_matrix: [16, 16]           - query weight matrix (int8)
// K_matrix: [16, 16]           - key weight matrix (int8)
// V_matrix: [16, 16]           - value weight matrix (int8)
// Q_output: [num_tokens, 16]   - query output (int8)
// K_output: [num_tokens, 16]   - key output (int8)
// V_output: [num_tokens, 16]   - value output (int8)
// 
// Each block processes 16 tokens
// Q = tokens @ Q_matrix
// K = tokens @ K_matrix
// V = tokens @ V_matrix

__global__ void compute_qkv(
    const i8* __restrict__ tokens,
    const i8* __restrict__ Q_matrix,
    const i8* __restrict__ K_matrix,
    const i8* __restrict__ V_matrix,
          i8* __restrict__ Q_output,
          i8* __restrict__ K_output,
          i8* __restrict__ V_output,
    const i32              num_tokens,
    const f32              scale
) {
    const i32 block_idx = blockIdx.x;
    const i32 token_off = block_idx * 256;  // 16 tokens * 16 dims
    const i32 tid       = threadIdx.x;
    
    const f32 scale_sq  = scale * scale;
    const f32 inv_scale = 1.0f / scale;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, i8, wmma::row_major> tok_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, i8, wmma::row_major> weight_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, i32> acc_frag;
    
    __shared__ i8  tok_smem[256];
    __shared__ i8  weight_smem[256];
    __shared__ i32 tmp_i32[256];
    
    // Load tokens block
    for (i32 i = tid; i < 256; i += 32) {
        tok_smem[i] = tokens[token_off + i];
    }
    __syncthreads();
    
    wmma::load_matrix_sync(tok_frag, tok_smem, 16);
    
    // Compute Q = tokens @ Q_matrix
    for (i32 i = tid; i < 256; i += 32) {
        weight_smem[i] = Q_matrix[i];
    }
    __syncthreads();
    
    wmma::load_matrix_sync(weight_frag, weight_smem, 16);
    wmma::fill_fragment(acc_frag, 0);
    wmma::mma_sync(acc_frag, tok_frag, weight_frag, acc_frag);
    wmma::store_matrix_sync(tmp_i32, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();
    
    for (i32 i = tid; i < 256; i += 32) {
        Q_output[token_off + i] = quantize(static_cast<f32>(tmp_i32[i]) * scale_sq, inv_scale);
    }
    __syncthreads();
    
    // Compute K = tokens @ K_matrix
    for (i32 i = tid; i < 256; i += 32) {
        weight_smem[i] = K_matrix[i];
    }
    __syncthreads();
    
    wmma::load_matrix_sync(weight_frag, weight_smem, 16);
    wmma::fill_fragment(acc_frag, 0);
    wmma::mma_sync(acc_frag, tok_frag, weight_frag, acc_frag);
    wmma::store_matrix_sync(tmp_i32, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();
    
    for (i32 i = tid; i < 256; i += 32) {
        K_output[token_off + i] = quantize(static_cast<f32>(tmp_i32[i]) * scale_sq, inv_scale);
    }
    __syncthreads();
    
    // Compute V = tokens @ V_matrix
    for (i32 i = tid; i < 256; i += 32) {
        weight_smem[i] = V_matrix[i];
    }
    __syncthreads();
    
    wmma::load_matrix_sync(weight_frag, weight_smem, 16);
    wmma::fill_fragment(acc_frag, 0);
    wmma::mma_sync(acc_frag, tok_frag, weight_frag, acc_frag);
    wmma::store_matrix_sync(tmp_i32, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();
    
    for (i32 i = tid; i < 256; i += 32) {
        V_output[token_off + i] = quantize(static_cast<f32>(tmp_i32[i]) * scale_sq, inv_scale);
    }
}

// Single head flash attention kernel for int8 Q, K, V with 16-dimensions per head
// num_q and num_kv must be multiples of 16
__global__ void flash_attention_int8_16(
    const i8* __restrict__ Q,
    const i8* __restrict__ K,
    const i8* __restrict__ V,
    i8* __restrict__       O,
    const i32              num_q,
    const i32              num_kv,
    const f32              scale
) {
    const i32 q_block   = blockIdx.x;
    const i32 q_off     = q_block * 256;
    const i32 tid       = threadIdx.x;
    
    const f32 scale_qk  = scale * scale;
    const f32 inv_scale = f32(1.0) / scale;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, i8, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, i8, wmma::row_major> k_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, i32> acc_frag;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> attn_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, f32> out_frag;
    
    __shared__ f32 row_max[16];
    __shared__ f32 row_sum[16];
    __shared__ i8  Q_smem[256];
    __shared__ i8  K_smem[256];
    __shared__ half V_smem[256];
    __shared__ half attn_smem[256];
    __shared__ i32 tmp_i32[256];
    __shared__ f32 scores[256];
    __shared__ f32 output[256];
    __shared__ f32 tmp_f32[256];
    
    if (tid < 16) {
        row_max[tid] = -1e10f;
        row_sum[tid] = 0.0f;
    }
    for (i32 i = tid; i < 256; i += 32) {
        output[i] = 0.0f;
        Q_smem[i] = Q[q_off + i];
    }
    __syncthreads();
    
    wmma::load_matrix_sync(q_frag, Q_smem, 16);
    
    const i32 num_kv_blocks = num_kv / 16;
    
    for (i32 kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const i32 kv_start = kv_block * 16;
        
        for (i32 i = tid; i < 256; i += 32) {
            i32 dim     = i / 16;
            i32 token   = i % 16;
            K_smem[i]   = K[(kv_start + token) * 16 + dim];
        }
        __syncthreads();
        
        wmma::load_matrix_sync(k_frag, K_smem, 16);
        wmma::fill_fragment(acc_frag, 0);
        wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);
        wmma::store_matrix_sync(tmp_i32, acc_frag, 16, wmma::mem_row_major);
        __syncthreads();
        
        for (i32 i = tid; i < 256; i += 32) {
            scores[i] = static_cast<f32>(tmp_i32[i]) * scale_qk;
        }
        __syncthreads();
        
        if (tid < 16) {
            f32 block_max = -1e10f;
            for (i32 j = 0; j < 16; j++) {
                block_max = fmaxf(block_max, scores[tid * 16 + j]);
            }
            
            f32 prev_max        = row_max[tid];
            f32 new_max         = fmaxf(prev_max, block_max);
            f32 correction      = expf(prev_max - new_max);
            
            row_sum[tid] *= correction;
            for (i32 j = 0; j < 16; j++) {
                output[tid * 16 + j] *= correction;
            }
            
            f32 block_sum = 0.0f;
            for (i32 j = 0; j < 16; j++) {
                f32 e                   = expf(scores[tid * 16 + j] - new_max);
                scores[tid * 16 + j]    = e;
                block_sum               += e;
            }
            
            row_max[tid] = new_max;
            row_sum[tid] += block_sum;
        }
        __syncthreads();
        
        for (i32 i = tid; i < 256; i += 32) {
            i32 token       = i / 16;
            i32 dim         = i % 16;
            attn_smem[i]    = __float2half(scores[i]);
            V_smem[i]       = __float2half(static_cast<f32>(V[(kv_start + token) * 16 + dim]) * scale);
        }
        __syncthreads();
        
        wmma::load_matrix_sync(attn_frag, attn_smem, 16);
        wmma::load_matrix_sync(v_frag, V_smem, 16);
        wmma::fill_fragment(out_frag, 0.0f);
        wmma::mma_sync(out_frag, attn_frag, v_frag, out_frag);
        wmma::store_matrix_sync(tmp_f32, out_frag, 16, wmma::mem_row_major);
        __syncthreads();
        
        for (i32 i = tid; i < 256; i += 32) {
            output[i] += tmp_f32[i];
        }
        __syncthreads();
    }
    
    if (tid < 16) {
        f32 inv_sum = 1.0f / row_sum[tid];
        for (i32 j = 0; j < 16; j++) {
            output[tid * 16 + j] *= inv_sum;
        }
    }
    __syncthreads();
    
    for (i32 i = tid; i < 256; i += 32) {
        O[q_off + i] = quantize(output[i], inv_scale);
    }
}

torch::Tensor flash_attention_int8(
    int64_t stream_ptr,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    double max_val
) {
    const i32 num_q = Q.size(0);
    const i32 num_kv = K.size(0);
    const f32 scale = static_cast<f32>(max_val) / f32(127.0);

    TORCH_CHECK(num_q % 16 == 0, "num_q must be multiple of 16");
    TORCH_CHECK(num_kv % 16 == 0, "num_kv must be multiple of 16");

    auto O = torch::empty({num_q, 16}, torch::dtype(torch::kInt8).device(Q.device()));
    
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    flash_attention_int8_16<<<num_q / 16, 32, 0, stream>>>(
        Q.data_ptr<i8>(),
        K.data_ptr<i8>(),
        V.data_ptr<i8>(),
        O.data_ptr<i8>(),
        num_q,
        num_kv,
        scale
    );
    
    return O;
}

torch::Tensor compute_qkv_fn(
    int64_t stream_ptr,
    torch::Tensor tokens,
    torch::Tensor Q_matrix,
    torch::Tensor K_matrix,
    torch::Tensor V_matrix,
    double max_val
) {
    TORCH_CHECK(tokens.size(1) == 16, "Token dim must be 16");
    TORCH_CHECK(tokens.size(0) % 16 == 0, "Token count must be multiple of 16");
    TORCH_CHECK(Q_matrix.size(0) == 16 && Q_matrix.size(1) == 16, "Q_matrix must be 16x16");
    TORCH_CHECK(K_matrix.size(0) == 16 && K_matrix.size(1) == 16, "K_matrix must be 16x16");
    TORCH_CHECK(V_matrix.size(0) == 16 && V_matrix.size(1) == 16, "V_matrix must be 16x16");
    
    const i32 num_tokens = tokens.size(0);
    const f32 scale = static_cast<f32>(max_val) / f32(127.0);
    
    auto Q_output = torch::empty({num_tokens, 16}, torch::dtype(torch::kInt8).device(tokens.device()));
    auto K_output = torch::empty({num_tokens, 16}, torch::dtype(torch::kInt8).device(tokens.device()));
    auto V_output = torch::empty({num_tokens, 16}, torch::dtype(torch::kInt8).device(tokens.device()));
    
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    compute_qkv<<<num_tokens / 16, 32, 0, stream>>>(
        tokens.data_ptr<i8>(),
        Q_matrix.data_ptr<i8>(),
        K_matrix.data_ptr<i8>(),
        V_matrix.data_ptr<i8>(),
        Q_output.data_ptr<i8>(),
        K_output.data_ptr<i8>(),
        V_output.data_ptr<i8>(),
        num_tokens,
        scale
    );
    
    return torch::stack({Q_output, K_output, V_output}, 0);  // [3, num_tokens, 16]
}

;//
"""

cpp_source = """
torch::Tensor flash_attention_int8(
    int64_t stream_ptr,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    double max_val);
torch::Tensor compute_qkv_fn(
    int64_t stream_ptr,
    torch::Tensor tokens,
    torch::Tensor Q_matrix,
    torch::Tensor K_matrix,
    torch::Tensor V_matrix,
    double max_val
);
"""

cuda_module = load_inline(
    name="cuda_module",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["flash_attention_int8", "compute_qkv_fn"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-arch=sm_80"]
)

def quantize(x, max_val):
    scale = max_val / 127.0
    return (x / scale).round().clamp(-127, 127).to(torch.int8)

def dequantize(q, max_val):
    return q.float() * (max_val / 127.0)

stream_0 = torch.cuda.Stream()

# torch.manual_seed(42)
num_q, num_kv = 16, 16
Q_f = torch.randn(num_q, 16, device="cuda")
K_f = torch.randn(num_kv, 16, device="cuda")
V_f = torch.randn(num_kv, 16, device="cuda")

max_val = 4.0

Q_i8 = quantize(Q_f, max_val)
K_i8 = quantize(K_f, max_val)
V_i8 = quantize(V_f, max_val)

Q_deq = dequantize(Q_i8, max_val)
K_deq = dequantize(K_i8, max_val)
V_deq = dequantize(V_i8, max_val)

O_ref = torch.softmax(Q_deq @ K_deq.T, dim=-1) @ V_deq

O_i8 = cuda_module.flash_attention_int8(stream_0.cuda_stream, Q_i8, K_i8, V_i8, max_val)
stream_0.synchronize()
O_deq = dequantize(O_i8, max_val)

print(f"max_val: {max_val}")
print(f"Output range ref: [{O_ref.min().item():.4f}, {O_ref.max().item():.4f}]")
print(f"Max error: {(O_deq - O_ref).abs().max().item():.6f}")
print(f"Mean error: {(O_deq - O_ref).abs().mean().item():.6f}")

print(f"\nRow 0 kernel: {O_deq[0, :4].tolist()}")
print(f"Row 0 ref:    {O_ref[0, :4].tolist()}")

# torch.manual_seed(42)
num_tokens  = 64

HEAD_DIM = 16

tokens_f = torch.randn(num_tokens, HEAD_DIM, device="cuda")
Q_mat_f  = torch.randn(16, HEAD_DIM, device="cuda") / math.sqrt(HEAD_DIM)
K_mat_f  = torch.randn(16, HEAD_DIM, device="cuda") / math.sqrt(HEAD_DIM)
V_mat_f  = torch.randn(16, HEAD_DIM, device="cuda") / math.sqrt(HEAD_DIM)

tokens_i8 = quantize(tokens_f, max_val)
Q_mat_i8  = quantize(Q_mat_f, max_val)
K_mat_i8  = quantize(K_mat_f, max_val)
V_mat_i8  = quantize(V_mat_f, max_val)

# Kernel
QKV = cuda_module.compute_qkv_fn(stream_0.cuda_stream,tokens_i8, Q_mat_i8, K_mat_i8, V_mat_i8, max_val)
stream_0.synchronize()
Q_out = dequantize(QKV[0], max_val)
K_out = dequantize(QKV[1], max_val)
V_out = dequantize(QKV[2], max_val)

# Reference
tokens_deq = dequantize(tokens_i8, max_val)
Q_mat_deq  = dequantize(Q_mat_i8, max_val)
K_mat_deq  = dequantize(K_mat_i8, max_val)
V_mat_deq  = dequantize(V_mat_i8, max_val)

Q_ref = tokens_deq @ Q_mat_deq
K_ref = tokens_deq @ K_mat_deq
V_ref = tokens_deq @ V_mat_deq

print(f"Q max error: {(Q_out - Q_ref).abs().max().item():.6f}")
print(f"K max error: {(K_out - K_ref).abs().max().item():.6f}")
print(f"V max error: {(V_out - V_ref).abs().max().item():.6f}")

print(f"\nQ row 0 kernel: {Q_out[0, :4].tolist()}")
print(f"Q row 0 ref:    {Q_ref[0, :4].tolist()}")
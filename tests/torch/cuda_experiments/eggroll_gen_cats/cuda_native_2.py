# MIT License
# Copyright (c) 2025 Anton Schreiner

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
//js

#define f32     float
#define i32     int32_t
#define i8      int8_t
#define u32     uint32_t
#define u64     uint64_t
#define INT8_MAX  127
#define INT8_MIN  -127 // For symmetry

#define ifor(N) for(i32 i=0;i<(N);i++)
#define jfor(N) for(i32 j=0;j<(N);j++)
#define kfor(N) for(i32 k=0;k<(N);k++)
#define xfor(N) for(i32 x=0;x<(N);x++)
#define yfor(N) for(i32 y=0;y<(N);y++)


#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

__device__ __forceinline__ i8 saturate_int8(int32_t val) {
    return static_cast<i8>(min(max(val, -128), 127));
}

__global__ void tensor_core_matmul_int8_saturate(
    const i8* __restrict__ A,
    const i8* __restrict__ B,
    i8* __restrict__ C
) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    
    wmma::fill_fragment(c_frag, 0);
    
    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);
    
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // Store to shared memory first, then convert
    __shared__ int32_t temp[16 * 16];
    wmma::store_matrix_sync(temp, c_frag, 16, wmma::mem_row_major);
    __syncthreads();
    
    // Each thread in the warp converts multiple elements
    const int tid = threadIdx.x;
    for (int i = tid; i < 256; i += 32) {
        C[i] = saturate_int8(temp[i]);
    }
}


// Symmetric quantization helpers
// q = round(x / scale), where scale = max_v / 127
// x = q * scale

__device__ __forceinline__ i8 quantize(f32 val, f32 scale) {
    f32 q = rintf(val / scale);
    return static_cast<i8>(min(max(q, -f32(INT8_MAX)), f32(INT8_MAX)));
}

__device__ __forceinline__ f32 dequantize(i32 q, f32 scale) {
    return static_cast<f32>(q) * scale;
}


// Flash Attention int8, head_dim=16
// Q, K, V, O all use same scale: scale = max_val / 127
// Token counts must be multiples of 16

__global__ void flash_attention_int8_16(
    const i8* __restrict__ Q,
    const i8* __restrict__ K,
    const i8* __restrict__ V,
    i8* __restrict__       O,
    const i32              num_q,
    const i32              num_kv,
    const f32              scale
) {
    const i32 q_block = blockIdx.x;
    const i32 q_off = q_block * 256;
    const i32 tid = threadIdx.x;
    
    const f32 scale_qk = scale * scale;
    const f32 inv_scale = 1.0f / scale;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, i8, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, i8, wmma::row_major> k_frag;  // Changed to row_major
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
        const i32 kv_off = kv_block * 256;
        
        // Load K^T: K is [num_kv, 16], we want [16, 16] block transposed
        // K[kv_start + row, col] -> K_smem[col, row] = K_smem[col * 16 + row]
        // For row_major fragment loading K^T:
        // We need K^T in row-major, i.e., K_smem[i] = K^T[i/16, i%16] = K[i%16, i/16]
        // But K is [num_kv, 16], so K[token_idx, dim] = K[token_idx * 16 + dim]
        // K^T[dim, token_idx] -> K_smem stored row-major: K_smem[dim * 16 + token_idx]
        for (i32 i = tid; i < 256; i += 32) {
            i32 dim = i / 16;        // row in K^T
            i32 token = i % 16;      // col in K^T
            // K[kv_start + token, dim]
            K_smem[i] = K[kv_off + token * 16 + dim];
        }
        __syncthreads();
        
        // Q @ K^T: Q[16,16] @ K^T[16,16] -> [16,16]
        wmma::load_matrix_sync(k_frag, K_smem, 16);
        wmma::fill_fragment(acc_frag, 0);
        wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);
        wmma::store_matrix_sync(tmp_i32, acc_frag, 16, wmma::mem_row_major);
        __syncthreads();
        
        for (i32 i = tid; i < 256; i += 32) {
            scores[i] = static_cast<f32>(tmp_i32[i]) * scale_qk;
        }
        __syncthreads();
        
        // Online softmax
        if (tid < 16) {
            f32 block_max = -1e10f;
            for (i32 j = 0; j < 16; j++) {
                block_max = fmaxf(block_max, scores[tid * 16 + j]);
            }
            
            f32 prev_max = row_max[tid];
            f32 new_max = fmaxf(prev_max, block_max);
            f32 correction = expf(prev_max - new_max);
            
            row_sum[tid] *= correction;
            for (i32 j = 0; j < 16; j++) {
                output[tid * 16 + j] *= correction;
            }
            
            f32 block_sum = 0.0f;
            for (i32 j = 0; j < 16; j++) {
                f32 e = expf(scores[tid * 16 + j] - new_max);
                scores[tid * 16 + j] = e;
                block_sum += e;
            }
            
            row_max[tid] = new_max;
            row_sum[tid] += block_sum;
        }
        __syncthreads();
        
        for (i32 i = tid; i < 256; i += 32) {
            attn_smem[i] = __float2half(scores[i]);
            V_smem[i] = __float2half(static_cast<f32>(V[kv_off + i]) * scale);
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
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    double max_val
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Must be CUDA");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "Must be contiguous");
    TORCH_CHECK(Q.scalar_type() == torch::kInt8, "Must be int8");
    TORCH_CHECK(Q.size(1) == 16 && K.size(1) == 16 && V.size(1) == 16, "Head dim must be 16");
    TORCH_CHECK(Q.size(0) % 16 == 0 && K.size(0) % 16 == 0, "Token count must be multiple of 16");
    TORCH_CHECK(K.size(0) == V.size(0), "K and V must match");
    
    const i32 num_q = Q.size(0);
    const i32 num_kv = K.size(0);
    const f32 scale = static_cast<f32>(max_val) / 127.0f;
    
    auto O = torch::empty({num_q, 16}, torch::dtype(torch::kInt8).device(Q.device()));
    
    flash_attention_int8_16<<<num_q / 16, 32>>>(
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

torch::Tensor matmul_int8_out(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(A.scalar_type() == torch::kInt8 && B.scalar_type() == torch::kInt8, "Must be int8");
    TORCH_CHECK(A.size(0) == 16 && A.size(1) == 16, "A must be 16x16");
    TORCH_CHECK(B.size(0) == 16 && B.size(1) == 16, "B must be 16x16");
    
    auto C = torch::empty({16, 16}, torch::dtype(torch::kInt8).device(A.device()));
    
    tensor_core_matmul_int8_saturate<<<1, 32>>>(
        A.data_ptr<int8_t>(),
        B.data_ptr<int8_t>(),
        C.data_ptr<int8_t>()
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

// Debug kernel - outputs f32 instead of i8, no final quantization
__global__ void flash_attention_debug(
    const i8* __restrict__ Q,
    const i8* __restrict__ K,
    const i8* __restrict__ V,
    f32* __restrict__      O,
    f32* __restrict__      debug_scores,
    const i32              num_q,
    const i32              num_kv,
    const f32              scale
) {
    const i32 q_block = blockIdx.x;
    const i32 q_off = q_block * 256;
    const i32 q_start = q_block * 16;
    const i32 tid = threadIdx.x;
    
    const f32 scale_qk = scale * scale;
    
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
        
        // Load K^T
        for (i32 i = tid; i < 256; i += 32) {
            i32 dim = i / 16;
            i32 token = i % 16;
            K_smem[i] = K[(kv_start + token) * 16 + dim];
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
        
        // Store debug scores
        for (i32 i = tid; i < 256; i += 32) {
            i32 q_idx = i / 16;
            i32 kv_idx = i % 16;
            debug_scores[(q_start + q_idx) * num_kv + kv_start + kv_idx] = scores[i];
        }
        __syncthreads();
        
        if (tid < 16) {
            f32 block_max = -1e10f;
            for (i32 j = 0; j < 16; j++) {
                block_max = fmaxf(block_max, scores[tid * 16 + j]);
            }
            
            f32 prev_max   = row_max[tid];
            f32 new_max    = fmaxf(prev_max, block_max);
            f32 correction = expf(prev_max - new_max);
            
            row_sum[tid] *= correction;
            for (i32 j = 0; j < 16; j++) {
                output[tid * 16 + j] *= correction;
            }
            
            f32 block_sum = 0.0f;
            for (i32 j = 0; j < 16; j++) {
                f32 e = expf(scores[tid * 16 + j] - new_max);
                scores[tid * 16 + j] = e;
                block_sum += e;
            }
            
            row_max[tid] = new_max;
            row_sum[tid] += block_sum;
        }
        __syncthreads();
        
        for (i32 i = tid; i < 256; i += 32) {
            i32 token = i / 16;
            i32 dim = i % 16;
            attn_smem[i] = __float2half(scores[i]);
            V_smem[i] = __float2half(static_cast<f32>(V[(kv_start + token) * 16 + dim]) * scale);
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
        O[q_off + i] = output[i];
    }
}

std::vector<torch::Tensor> flash_attention_debug_fn(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    double max_val
) {
    const i32 num_q = Q.size(0);
    const i32 num_kv = K.size(0);
    const f32 scale = static_cast<f32>(max_val) / 127.0f;
    
    auto O = torch::empty({num_q, 16}, torch::dtype(torch::kFloat32).device(Q.device()));
    auto debug_scores = torch::empty({num_q, num_kv}, torch::dtype(torch::kFloat32).device(Q.device()));
    
    flash_attention_debug<<<num_q / 16, 32>>>(
        Q.data_ptr<i8>(),
        K.data_ptr<i8>(),
        V.data_ptr<i8>(),
        O.data_ptr<f32>(),
        debug_scores.data_ptr<f32>(),
        num_q,
        num_kv,
        scale
    );
    
    return {O, debug_scores};
}

;//
"""

cpp_source = """
torch::Tensor flash_attention_int8(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double max_val);
std::vector<torch::Tensor> flash_attention_debug_fn(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double max_val);
"""

flash_attn = load_inline(
    name="flash_attn_int8_debug",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["flash_attention_int8", "flash_attention_debug_fn"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-arch=sm_80"]
)

def quantize(x, max_val):
    scale = max_val / 127.0
    return (x / scale).round().clamp(-127, 127).to(torch.int8)

def dequantize(q, max_val):
    return q.float() * (max_val / 127.0)

torch.manual_seed(42)
num_q, num_kv = 64, 128
Q_f = torch.randn(num_q, 16, device="cuda")
K_f = torch.randn(num_kv, 16, device="cuda")
V_f = torch.randn(num_kv, 16, device="cuda")

max_val = 3.0 # max(Q_f.abs().max(), K_f.abs().max(), V_f.abs().max()).item()

Q_i8 = quantize(Q_f, max_val)
K_i8 = quantize(K_f, max_val)
V_i8 = quantize(V_f, max_val)

Q_deq = dequantize(Q_i8, max_val)
K_deq = dequantize(K_i8, max_val)
V_deq = dequantize(V_i8, max_val)

O_f32, debug_scores = flash_attn.flash_attention_debug_fn(Q_i8, K_i8, V_i8, max_val)

scores_ref = Q_deq @ K_deq.T
attn_ref = torch.softmax(scores_ref, dim=-1)
O_ref = attn_ref @ V_deq

print("=== Step-by-step comparison ===")
print(f"Scores error: {(debug_scores - scores_ref).abs().max().item():.6f}")
print(f"Scores range kernel: [{debug_scores.min().item():.2f}, {debug_scores.max().item():.2f}]")
print(f"Scores range ref: [{scores_ref.min().item():.2f}, {scores_ref.max().item():.2f}]")

attn_from_kernel = torch.softmax(debug_scores, dim=-1)
print(f"Attn from kernel scores error: {(attn_from_kernel - attn_ref).abs().max().item():.6f}")

print(f"\nOutput error (f32): {(O_f32 - O_ref).abs().max().item():.6f}")
print(f"Output mean error: {(O_f32 - O_ref).abs().mean().item():.6f}")

print(f"\nRow 0 kernel: {O_f32[0, :4].tolist()}")
print(f"Row 0 ref:    {O_ref[0, :4].tolist()}")

# Also test final int8 version
O_i8 = flash_attn.flash_attention_int8(Q_i8, K_i8, V_i8, max_val)
O_deq = dequantize(O_i8, max_val)
print(f"\nFinal int8 output error: {(O_deq - O_ref).abs().max().item():.4f}")
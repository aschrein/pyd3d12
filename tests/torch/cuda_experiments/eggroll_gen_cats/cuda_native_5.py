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

__device__ __forceinline__ i8 quantize(f32 val, f32 inv_scale) {
    return static_cast<i8>(min(max(rintf(val * inv_scale), -127.0f), 127.0f));
}

// ============ Patchifier ============
// Input:  images [batch, 64, 64, 3] - HWC layout
// Output: tokens [batch, 64, num_heads * 16]
//
// Each patch is 8x8x3 = 192 pixels -> project to num_heads * 16 dims
// weights: [12 * num_heads, 16, 16] for wmma
//
// Grid: (4, batch) - 4 groups of 16 patches

__global__ void patchify_and_project(
    const f32* __restrict__ images,        // [batch, 64, 64, 3]
    const i8*  __restrict__ proj_weights,  // [12 * num_heads, 16, 16]
          i8*  __restrict__ tokens,        // [batch, 64, num_heads * 16]
    const i32              batch_size,
    const i32              num_heads,
    const f32              scale
) {
    const i32 patch_group = blockIdx.x;
    const i32 batch_idx   = blockIdx.y;
    const i32 tid         = threadIdx.x;
    
    const f32 scale_sq = scale * scale;
    const f32 inv_scale = 1.0f / scale;
    
    const i32 patch_start = patch_group * 16;
    const i32 out_dim = num_heads * 16;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, i8, wmma::row_major> patch_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, i8, wmma::row_major> weight_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, i32> acc_frag;
    
    __shared__ i8  patch_smem[256];
    __shared__ i8  weight_smem[256];
    __shared__ i32 tmp_i32[256];
    __shared__ f32 accumulated[256];
    
    for (i32 out_chunk = 0; out_chunk < num_heads; out_chunk++) {
        
        for (i32 i = tid; i < 256; i += 32) {
            accumulated[i] = 0.0f;
        }
        __syncthreads();
        
        // Process 192 input dims in chunks of 16 (12 chunks)
        // Layout: first 64 values = 8x8 R, next 64 = G, next 64 = B
        // But with HWC, each pixel has 3 consecutive channels
        // Chunk layout: chunk 0-3 = R channel positions, 4-7 = G, 8-11 = B
        for (i32 in_chunk = 0; in_chunk < 12; in_chunk++) {
            const i32 channel = in_chunk / 4;
            const i32 chunk_in_channel = in_chunk % 4;
            
            for (i32 i = tid; i < 256; i += 32) {
                const i32 local_patch = i / 16;
                const i32 local_dim = i % 16;
                
                const i32 patch_idx = patch_start + local_patch;
                const i32 py = patch_idx / 8;
                const i32 px = patch_idx % 8;
                
                const i32 pos_in_patch = chunk_in_channel * 16 + local_dim;
                const i32 local_y = pos_in_patch / 8;
                const i32 local_x = pos_in_patch % 8;
                
                const i32 img_y = py * 8 + local_y;
                const i32 img_x = px * 8 + local_x;
                
                // HWC layout: images[batch, y, x, c]
                const f32 pixel = images[batch_idx * 64 * 64 * 3 + img_y * 64 * 3 + img_x * 3 + channel];
                patch_smem[i] = quantize(pixel, inv_scale);
            }
            
            const i32 weight_idx = (in_chunk * num_heads + out_chunk) * 256;
            for (i32 i = tid; i < 256; i += 32) {
                weight_smem[i] = proj_weights[weight_idx + i];
            }
            __syncthreads();
            
            wmma::load_matrix_sync(patch_frag, patch_smem, 16);
            wmma::load_matrix_sync(weight_frag, weight_smem, 16);
            wmma::fill_fragment(acc_frag, 0);
            wmma::mma_sync(acc_frag, patch_frag, weight_frag, acc_frag);
            wmma::store_matrix_sync(tmp_i32, acc_frag, 16, wmma::mem_row_major);
            __syncthreads();
            
            for (i32 i = tid; i < 256; i += 32) {
                accumulated[i] += static_cast<f32>(tmp_i32[i]) * scale_sq;
            }
            __syncthreads();
        }
        
        const i32 token_offset = batch_idx * 64 * out_dim + patch_start * out_dim + out_chunk * 16;
        for (i32 i = tid; i < 256; i += 32) {
            const i32 local_patch = i / 16;
            const i32 local_dim = i % 16;
            tokens[token_offset + local_patch * out_dim + local_dim] = quantize(accumulated[i], inv_scale);
        }
        __syncthreads();
    }
}
// ============ Final Projection ============
// Input:  tokens [batch, 64, 16]
// Output: output [batch, 64, 64] -> reshape to [batch, 64, 8, 8]
//
// Grid: (4, batch) - 4 groups of 16 tokens

__global__ void final_projection(
    const i8* __restrict__ tokens,
    const i8* __restrict__ proj_weights,  // [4, 16, 16]
          i8* __restrict__ output,
    const i32              batch_size,
    const f32              scale
) {
    const i32 token_group = blockIdx.x;
    const i32 batch_idx   = blockIdx.y;
    const i32 tid         = threadIdx.x;
    
    const f32 scale_sq = scale * scale;
    const f32 inv_scale = 1.0f / scale;
    
    const i32 token_start = token_group * 16;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, i8, wmma::row_major> tok_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, i8, wmma::row_major> weight_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, i32> acc_frag;
    
    __shared__ i8  tok_smem[256];
    __shared__ i8  weight_smem[256];
    __shared__ i32 tmp_i32[256];
    
    const i32 token_offset = batch_idx * 64 * 16 + token_start * 16;
    for (i32 i = tid; i < 256; i += 32) {
        tok_smem[i] = tokens[token_offset + i];
    }
    __syncthreads();
    
    wmma::load_matrix_sync(tok_frag, tok_smem, 16);
    
    for (i32 out_chunk = 0; out_chunk < 4; out_chunk++) {
        for (i32 i = tid; i < 256; i += 32) {
            weight_smem[i] = proj_weights[out_chunk * 256 + i];
        }
        __syncthreads();
        
        wmma::load_matrix_sync(weight_frag, weight_smem, 16);
        wmma::fill_fragment(acc_frag, 0);
        wmma::mma_sync(acc_frag, tok_frag, weight_frag, acc_frag);
        wmma::store_matrix_sync(tmp_i32, acc_frag, 16, wmma::mem_row_major);
        __syncthreads();
        
        const i32 out_offset = batch_idx * 64 * 64 + token_start * 64 + out_chunk * 16;
        for (i32 i = tid; i < 256; i += 32) {
            const i32 local_tok = i / 16;
            const i32 local_dim = i % 16;
            output[out_offset + local_tok * 64 + local_dim] = quantize(static_cast<f32>(tmp_i32[i]) * scale_sq, inv_scale);
        }
        __syncthreads();
    }
}

// ============ Attention Kernels ============

__global__ void compute_qkv_batched(
    const i8* __restrict__ tokens,
    const i8* __restrict__ QKV_weights,
          i8* __restrict__ Q_output,
          i8* __restrict__ K_output,
          i8* __restrict__ V_output,
    const i32              batch_size,
    const i32              num_tokens,
    const i32              num_heads,
    const f32              scale
) {
    const i32 token_block = blockIdx.x;
    const i32 head_idx    = blockIdx.y;
    const i32 batch_idx   = blockIdx.z;
    const i32 tid         = threadIdx.x;
    
    const f32 scale_sq  = scale * scale;
    const f32 inv_scale = 1.0f / scale;
    
    const i32 tokens_offset = batch_idx * num_tokens * 16 + token_block * 256;
    const i32 output_offset = batch_idx * num_heads * num_tokens * 16 
                            + head_idx * num_tokens * 16 
                            + token_block * 256;
    const i32 weight_offset = head_idx * 3 * 256;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, i8, wmma::row_major> tok_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, i8, wmma::row_major> weight_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, i32> acc_frag;
    
    __shared__ i8  tok_smem[256];
    __shared__ i8  weight_smem[256];
    __shared__ i32 tmp_i32[256];
    
    for (i32 i = tid; i < 256; i += 32) {
        tok_smem[i] = tokens[tokens_offset + i];
    }
    __syncthreads();
    
    wmma::load_matrix_sync(tok_frag, tok_smem, 16);
    
    // Q
    for (i32 i = tid; i < 256; i += 32) {
        weight_smem[i] = QKV_weights[weight_offset + i];
    }
    __syncthreads();
    
    wmma::load_matrix_sync(weight_frag, weight_smem, 16);
    wmma::fill_fragment(acc_frag, 0);
    wmma::mma_sync(acc_frag, tok_frag, weight_frag, acc_frag);
    wmma::store_matrix_sync(tmp_i32, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();
    
    for (i32 i = tid; i < 256; i += 32) {
        Q_output[output_offset + i] = quantize(static_cast<f32>(tmp_i32[i]) * scale_sq, inv_scale);
    }
    __syncthreads();
    
    // K
    for (i32 i = tid; i < 256; i += 32) {
        weight_smem[i] = QKV_weights[weight_offset + 256 + i];
    }
    __syncthreads();
    
    wmma::load_matrix_sync(weight_frag, weight_smem, 16);
    wmma::fill_fragment(acc_frag, 0);
    wmma::mma_sync(acc_frag, tok_frag, weight_frag, acc_frag);
    wmma::store_matrix_sync(tmp_i32, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();
    
    for (i32 i = tid; i < 256; i += 32) {
        K_output[output_offset + i] = quantize(static_cast<f32>(tmp_i32[i]) * scale_sq, inv_scale);
    }
    __syncthreads();
    
    // V
    for (i32 i = tid; i < 256; i += 32) {
        weight_smem[i] = QKV_weights[weight_offset + 512 + i];
    }
    __syncthreads();
    
    wmma::load_matrix_sync(weight_frag, weight_smem, 16);
    wmma::fill_fragment(acc_frag, 0);
    wmma::mma_sync(acc_frag, tok_frag, weight_frag, acc_frag);
    wmma::store_matrix_sync(tmp_i32, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();
    
    for (i32 i = tid; i < 256; i += 32) {
        V_output[output_offset + i] = quantize(static_cast<f32>(tmp_i32[i]) * scale_sq, inv_scale);
    }
}

__global__ void flash_attention_batched(
    const i8* __restrict__ Q,
    const i8* __restrict__ K,
    const i8* __restrict__ V,
          i8* __restrict__ O,
    const i32              batch_size,
    const i32              num_heads,
    const i32              num_q,
    const i32              num_kv,
    const f32              scale
) {
    const i32 q_block   = blockIdx.x;
    const i32 head_idx  = blockIdx.y;
    const i32 batch_idx = blockIdx.z;
    const i32 tid       = threadIdx.x;
    
    const f32 scale_qk  = scale * scale;
    const f32 inv_scale = 1.0f / scale;
    
    const i32 qo_offset = batch_idx * num_heads * num_q * 16 
                        + head_idx * num_q * 16 
                        + q_block * 256;
    const i32 kv_base   = batch_idx * num_heads * num_kv * 16 
                        + head_idx * num_kv * 16;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, i8, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, i8, wmma::row_major> k_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, i32> acc_frag;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> attn_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, f32> out_frag;
    
    __shared__ f32  row_max[16];
    __shared__ f32  row_sum[16];
    __shared__ i8   Q_smem[256];
    __shared__ i8   K_smem[256];
    __shared__ half V_smem[256];
    __shared__ half attn_smem[256];
    __shared__ i32  tmp_i32[256];
    __shared__ f32  scores[256];
    __shared__ f32  output[256];
    __shared__ f32  tmp_f32[256];
    
    if (tid < 16) {
        row_max[tid] = -1e10f;
        row_sum[tid] = 0.0f;
    }
    for (i32 i = tid; i < 256; i += 32) {
        output[i] = 0.0f;
        Q_smem[i] = Q[qo_offset + i];
    }
    __syncthreads();
    
    wmma::load_matrix_sync(q_frag, Q_smem, 16);
    
    const i32 num_kv_blocks = num_kv / 16;
    
    for (i32 kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const i32 kv_offset = kv_base + kv_block * 256;
        
        for (i32 i = tid; i < 256; i += 32) {
            i32 dim   = i / 16;
            i32 token = i % 16;
            K_smem[i] = K[kv_offset + token * 16 + dim];
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
            i32 token    = i / 16;
            i32 dim      = i % 16;
            attn_smem[i] = __float2half(scores[i]);
            V_smem[i]    = __float2half(static_cast<f32>(V[kv_offset + token * 16 + dim]) * scale);
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
        O[qo_offset + i] = quantize(output[i], inv_scale);
    }
}

__global__ void project_and_residual(
    const i8* __restrict__ attn_output,
    const i8* __restrict__ proj_weights,
    const i8* __restrict__ residual,
          i8* __restrict__ output,
    const i32              batch_size,
    const i32              num_heads,
    const i32              num_tokens,
    const f32              scale
) {
    const i32 token_block = blockIdx.x;
    const i32 batch_idx   = blockIdx.y;
    const i32 tid         = threadIdx.x;
    
    const f32 scale_sq  = scale * scale;
    const f32 inv_scale = 1.0f / scale;
    
    const i32 residual_offset = batch_idx * num_tokens * 16 + token_block * 256;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, i8, wmma::row_major> attn_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, i8, wmma::row_major> proj_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, i32> acc_frag;
    
    __shared__ i8  attn_smem[256];
    __shared__ i8  proj_smem[256];
    __shared__ i32 tmp_i32[256];
    __shared__ f32 accumulated[256];
    
    for (i32 i = tid; i < 256; i += 32) {
        accumulated[i] = static_cast<f32>(residual[residual_offset + i]) * scale;
    }
    __syncthreads();
    
    for (i32 head_idx = 0; head_idx < num_heads; head_idx++) {
        const i32 attn_offset = batch_idx * num_heads * num_tokens * 16 
                              + head_idx * num_tokens * 16 
                              + token_block * 256;
        const i32 proj_offset = head_idx * 256;
        
        for (i32 i = tid; i < 256; i += 32) {
            attn_smem[i] = attn_output[attn_offset + i];
            proj_smem[i] = proj_weights[proj_offset + i];
        }
        __syncthreads();
        
        wmma::load_matrix_sync(attn_frag, attn_smem, 16);
        wmma::load_matrix_sync(proj_frag, proj_smem, 16);
        wmma::fill_fragment(acc_frag, 0);
        wmma::mma_sync(acc_frag, attn_frag, proj_frag, acc_frag);
        wmma::store_matrix_sync(tmp_i32, acc_frag, 16, wmma::mem_row_major);
        __syncthreads();
        
        for (i32 i = tid; i < 256; i += 32) {
            accumulated[i] += static_cast<f32>(tmp_i32[i]) * scale_sq;
        }
        __syncthreads();
    }
    
    for (i32 i = tid; i < 256; i += 32) {
        output[residual_offset + i] = quantize(accumulated[i], inv_scale);
    }
}

// ============ Host functions ============

torch::Tensor patchify_fn(
    int64_t stream_ptr,
    torch::Tensor images,
    torch::Tensor proj_weights,
    double max_val
) {
    TORCH_CHECK(images.dim() == 4, "images must be [batch, 64, 64, 3]");
    TORCH_CHECK(images.size(1) == 64 && images.size(2) == 64 && images.size(3) == 3, 
                "images must be [batch, 64, 64, 3]");
    TORCH_CHECK(proj_weights.dim() == 3, "proj_weights must be [12*num_heads, 16, 16]");
    TORCH_CHECK(proj_weights.size(0) % 12 == 0, "proj_weights.size(0) must be multiple of 12");
    
    const i32 batch_size = images.size(0);
    const i32 num_heads = proj_weights.size(0) / 12;
    const i32 out_dim = num_heads * 16;
    const f32 scale = static_cast<f32>(max_val) / 127.0f;
    
    auto tokens = torch::empty({batch_size, 64, out_dim}, 
                               torch::dtype(torch::kInt8).device(images.device()));
    
    dim3 grid(4, batch_size);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    patchify_and_project<<<grid, 32, 0, stream>>>(
        images.data_ptr<f32>(),
        proj_weights.data_ptr<i8>(),
        tokens.data_ptr<i8>(),
        batch_size,
        num_heads,
        scale
    );
    
    return tokens;
}

torch::Tensor final_projection_fn(
    int64_t stream_ptr,
    torch::Tensor tokens,
    torch::Tensor proj_weights,
    double max_val
) {
    TORCH_CHECK(tokens.dim() == 3, "tokens must be [batch, 64, 16]");
    TORCH_CHECK(tokens.size(1) == 64 && tokens.size(2) == 16, "tokens must be [batch, 64, 16]");
    TORCH_CHECK(proj_weights.dim() == 3, "proj_weights must be [4, 16, 16]");
    
    const i32 batch_size = tokens.size(0);
    const f32 scale = static_cast<f32>(max_val) / 127.0f;
    
    auto output = torch::empty({batch_size, 64, 64}, 
                               torch::dtype(torch::kInt8).device(tokens.device()));
    
    dim3 grid(4, batch_size);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    final_projection<<<grid, 32, 0, stream>>>(
        tokens.data_ptr<i8>(),
        proj_weights.data_ptr<i8>(),
        output.data_ptr<i8>(),
        batch_size,
        scale
    );
    
    return output;
}

std::vector<torch::Tensor> compute_qkv_batched_fn(
    int64_t stream_ptr,
    torch::Tensor tokens,
    torch::Tensor QKV_weights,
    double max_val
) {
    TORCH_CHECK(tokens.dim() == 3, "tokens must be [batch, num_tokens, 16]");
    TORCH_CHECK(tokens.size(2) == 16, "Token dim must be 16");
    TORCH_CHECK(tokens.size(1) % 16 == 0, "num_tokens must be multiple of 16");
    TORCH_CHECK(QKV_weights.dim() == 4, "QKV_weights must be [num_heads, 3, 16, 16]");
    
    const i32 batch_size = tokens.size(0);
    const i32 num_tokens = tokens.size(1);
    const i32 num_heads  = QKV_weights.size(0);
    const f32 scale      = static_cast<f32>(max_val) / 127.0f;
    
    auto Q_out = torch::empty({batch_size, num_heads, num_tokens, 16}, 
                              torch::dtype(torch::kInt8).device(tokens.device()));
    auto K_out = torch::empty({batch_size, num_heads, num_tokens, 16}, 
                              torch::dtype(torch::kInt8).device(tokens.device()));
    auto V_out = torch::empty({batch_size, num_heads, num_tokens, 16}, 
                              torch::dtype(torch::kInt8).device(tokens.device()));
    
    dim3 grid(num_tokens / 16, num_heads, batch_size);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    compute_qkv_batched<<<grid, 32, 0, stream>>>(
        tokens.data_ptr<i8>(),
        QKV_weights.data_ptr<i8>(),
        Q_out.data_ptr<i8>(),
        K_out.data_ptr<i8>(),
        V_out.data_ptr<i8>(),
        batch_size,
        num_tokens,
        num_heads,
        scale
    );
    
    return {Q_out, K_out, V_out};
}

torch::Tensor flash_attention_batched_fn(
    int64_t stream_ptr,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    double max_val
) {
    TORCH_CHECK(Q.dim() == 4, "Q must be [batch, num_heads, num_q, 16]");
    TORCH_CHECK(Q.size(3) == 16, "Head dim must be 16");
    TORCH_CHECK(Q.size(2) % 16 == 0, "num_q must be multiple of 16");
    TORCH_CHECK(K.size(2) % 16 == 0, "num_kv must be multiple of 16");
    
    const i32 batch_size = Q.size(0);
    const i32 num_heads  = Q.size(1);
    const i32 num_q      = Q.size(2);
    const i32 num_kv     = K.size(2);
    const f32 scale      = static_cast<f32>(max_val) / 127.0f;
    
    auto O = torch::empty({batch_size, num_heads, num_q, 16}, 
                          torch::dtype(torch::kInt8).device(Q.device()));
    
    dim3 grid(num_q / 16, num_heads, batch_size);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    flash_attention_batched<<<grid, 32, 0, stream>>>(
        Q.data_ptr<i8>(),
        K.data_ptr<i8>(),
        V.data_ptr<i8>(),
        O.data_ptr<i8>(),
        batch_size,
        num_heads,
        num_q,
        num_kv,
        scale
    );
    
    return O;
}

torch::Tensor project_and_residual_fn(
    int64_t stream_ptr,
    torch::Tensor attn_output,
    torch::Tensor proj_weights,
    torch::Tensor residual,
    double max_val
) {
    TORCH_CHECK(attn_output.dim() == 4, "attn_output must be [batch, num_heads, num_tokens, 16]");
    TORCH_CHECK(proj_weights.dim() == 3, "proj_weights must be [num_heads, 16, 16]");
    TORCH_CHECK(residual.dim() == 3, "residual must be [batch, num_tokens, 16]");
    TORCH_CHECK(attn_output.size(3) == 16, "Head dim must be 16");
    TORCH_CHECK(attn_output.size(2) % 16 == 0, "num_tokens must be multiple of 16");
    
    const i32 batch_size = attn_output.size(0);
    const i32 num_heads  = attn_output.size(1);
    const i32 num_tokens = attn_output.size(2);
    const f32 scale      = static_cast<f32>(max_val) / 127.0f;
    
    auto output = torch::empty({batch_size, num_tokens, 16}, 
                               torch::dtype(torch::kInt8).device(attn_output.device()));
    
    dim3 grid(num_tokens / 16, batch_size);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    project_and_residual<<<grid, 32, 0, stream>>>(
        attn_output.data_ptr<i8>(),
        proj_weights.data_ptr<i8>(),
        residual.data_ptr<i8>(),
        output.data_ptr<i8>(),
        batch_size,
        num_heads,
        num_tokens,
        scale
    );
    
    return output;
}
;//
"""

cpp_source = """
//js
torch::Tensor patchify_fn(int64_t stream_ptr, torch::Tensor images, torch::Tensor proj_weights, double max_val);
torch::Tensor final_projection_fn(int64_t stream_ptr, torch::Tensor tokens, torch::Tensor proj_weights, double max_val);
std::vector<torch::Tensor> compute_qkv_batched_fn(int64_t stream_ptr, torch::Tensor tokens, torch::Tensor QKV_weights, double max_val);
torch::Tensor flash_attention_batched_fn(int64_t stream_ptr, torch::Tensor Q, torch::Tensor K, torch::Tensor V, double max_val);
torch::Tensor project_and_residual_fn(int64_t stream_ptr, torch::Tensor attn_output, torch::Tensor proj_weights, torch::Tensor residual, double max_val);
;//
"""

module = load_inline(
    name="vit_int8_simple",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["patchify_fn", "final_projection_fn", "compute_qkv_batched_fn", 
               "flash_attention_batched_fn", "project_and_residual_fn"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-arch=sm_80"]
)

def quantize(x, max_val):
    scale = max_val / 127.0
    return (x / scale).round().clamp(-127, 127).to(torch.int8)

def dequantize(q, max_val):
    return q.float() * (max_val / 127.0)

# ============ Test Full Pipeline ============
print("=" * 60)
print("Testing Full ViT Pipeline (Single Scale)")
print("=" * 60)

batch_size = 4
num_heads = 1
num_tokens = 64
head_dim = 16
max_val = 6.0
num_attention_layers = 4

torch.manual_seed(123)
stream = torch.cuda.Stream()

images = torch.randn(batch_size, 64, 64, 3, device="cuda") / math.sqrt(3)

# Patchifier weights: [12, 16, 16] 12 groups of 16x16 weights
patchifier_weights_f = torch.randn(12, 16, 16, device="cuda") / math.sqrt(192)
patchifier_weights_i8 = quantize(patchifier_weights_f, max_val)

# Final projection weights: [4, 16, 16]
final_proj_weights_f = torch.randn(4, 16, 16, device="cuda") / math.sqrt(16)
final_proj_weights_i8 = quantize(final_proj_weights_f, max_val)

# Attention layers
class AttentionLayer:
    def __init__(self, head_dim, num_heads, max_val):
        QKV_weights_f = torch.randn(num_heads, 3, head_dim, head_dim, device="cuda") / math.sqrt(head_dim)
        self.QKV_weights_i8 = quantize(QKV_weights_f, max_val)
        proj_weights_f = torch.randn(num_heads, head_dim, head_dim, device="cuda") / math.sqrt(head_dim * num_heads)
        self.proj_weights_i8 = quantize(proj_weights_f, max_val)

attention_layers = [AttentionLayer(head_dim, num_heads, max_val) for _ in range(num_attention_layers)]

# ============ Run Pipeline ============
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record(stream)

# 1. Patchify
tokens_i8 = module.patchify_fn(stream.cuda_stream, images, patchifier_weights_i8, max_val)

# 2. Attention layers
x_i8 = tokens_i8
for layer in attention_layers:
    Q_i8, K_i8, V_i8 = module.compute_qkv_batched_fn(stream.cuda_stream, x_i8, layer.QKV_weights_i8, max_val)
    attn_out_i8 = module.flash_attention_batched_fn(stream.cuda_stream, Q_i8, K_i8, V_i8, max_val)
    x_i8 = module.project_and_residual_fn(stream.cuda_stream, attn_out_i8, layer.proj_weights_i8, x_i8, max_val)

# 3. Final projection
output_i8 = module.final_projection_fn(stream.cuda_stream, x_i8, final_proj_weights_i8, max_val)

end_event.record(stream)
end_event.synchronize()

elapsed_ms = start_event.elapsed_time(end_event)

print(f"\nPipeline:")
print(f"  Input:  images {list(images.shape)}")
print(f"  After patchify: tokens {list(tokens_i8.shape)}")
print(f"  After {num_attention_layers} attention layers: {list(x_i8.shape)}")
print(f"  Output: {list(output_i8.shape)}")
print(f"\nElapsed time: {elapsed_ms:.3f} ms")

# Reshape output to [batch, 64, 8, 8]
output_reshaped = output_i8.view(batch_size, 64, 8, 8)
print(f"Output reshaped: {list(output_reshaped.shape)}")

# ============ Verify Patchifier ============
print("\n" + "=" * 60)
print("Verifying Patchifier")
print("=" * 60)

from einops import rearrange

def patchify_reference(images, weights, max_val):
    """
    images: [batch, 64, 64, 3] - HWC layout
    weights: [12*num_heads, 16, 16]
    """
    batch = images.shape[0]
    num_heads = weights.shape[0] // 12
    out_dim = num_heads * 16
    
    # Extract patches with channels last then reorder to channels-grouped
    # [batch, 64, 64, 3] -> [batch, 8, 8, 8, 8, 3] -> [batch, 64, 192] with (c p1 p2) order
    patches = rearrange(images, 'b (h p1) (w p2) c -> b (h w) (c p1 p2)', p1=8, p2=8)
    
    patches_q = quantize(patches, max_val)
    patches_deq = dequantize(patches_q, max_val)
    
    weights_deq = dequantize(weights, max_val)
    weights_flat = rearrange(weights_deq, '(i n) s1 s2 -> (i s1) (n s2)', i=12, n=num_heads)
    
    output = patches_deq @ weights_flat
    return output

# Test:
images = torch.randn(batch_size, 64, 64, 3, device="cuda") # HWC layout

tokens_i8 = module.patchify_fn(stream.cuda_stream, images, patchifier_weights_i8, max_val)
stream.synchronize()

tokens_ref = patchify_reference(images, patchifier_weights_i8, max_val)
tokens_kernel = dequantize(tokens_i8, max_val)

print(f"Tokens shape: {tokens_kernel.shape}")
print(f"Max error: {(tokens_kernel - tokens_ref).abs().max().item():.6f}")
print(f"Mean error: {(tokens_kernel - tokens_ref).abs().mean().item():.6f}")

# ============ Verify Final Projection ============
print("\n" + "=" * 60)
print("Verifying Final Projection")
print("=" * 60)

def final_proj_reference(tokens, weights, max_val):
    tokens_deq = dequantize(tokens, max_val)
    weights_deq = dequantize(weights, max_val)
    weights_flat = weights_deq.permute(1, 0, 2).reshape(16, 64)
    output = tokens_deq @ weights_flat
    return output

output_ref = final_proj_reference(x_i8, final_proj_weights_i8, max_val)
output_kernel = dequantize(output_i8, max_val)

print(f"Output shape: {output_kernel.shape}")
print(f"Max error: {(output_kernel - output_ref).abs().max().item():.6f}")
print(f"Mean error: {(output_kernel - output_ref).abs().mean().item():.6f}")

print("\n" + "=" * 60)
print("Full ViT Pipeline Test Complete!")
print("=" * 60)
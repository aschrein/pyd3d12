# MIT License
# Copyright (c) 2025 Anton Schreiner

import torch
import torch.utils.cpp_extension
import time
import math
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import random

cuda_source = """
//js
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

using f32 = float;
using f16 = half;
using i32 = int32_t;
using u32 = uint32_t;

// ============ Random Number Generation ============

__device__ __forceinline__ u32 lowbias32(u32 x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ f32 random_normal(u32 seed) {
    u32 s1 = lowbias32(seed);
    u32 s2 = lowbias32(s1);
    
    f32 u1 = fmaxf((f32)(s1 & 0xFFFF) / 65535.0f, 1e-7f);
    f32 u2 = (f32)(s2 & 0xFFFF) / 65535.0f;
    
    f32 radius = sqrtf(-2.0f * logf(u1));
    f32 theta = 2.0f * 3.14159265358979323846f * u2;
    return radius * cosf(theta);
}

// EGGROLL-style rank-1 perturbation: E = (1/sqrt(r)) * A @ B.T
// For rank-1: E[row, col] = A[row] * B[col] / sqrt(1) = A[row] * B[col]
// A and B are generated from independent seeds per (batch, layer)
__device__ __forceinline__ f32 get_perturbation_eggroll(
    u32 base_seed, i32 batch_idx, u32 layer_id, i32 row, i32 col, f32 sigma
) {
    // Generate A[row] - seed depends on (base_seed, batch_idx, layer_id, row)
    u32 a_seed = base_seed ^ (batch_idx * 12345u) ^ (layer_id * 111111u) ^ (row * 67890u);
    f32 a_val = random_normal(a_seed);
    
    // Generate B[col] - different mixing to ensure independence from A
    u32 b_seed = base_seed ^ (batch_idx * 54321u) ^ (layer_id * 222222u) ^ ((col + 500000) * 98765u);
    f32 b_val = random_normal(b_seed);
    
    // Rank-1 outer product element: A[row] * B[col] * sigma
    // No additional normalization needed - the paper shows rank-1 works well
    f32 p = sigma * a_val * b_val;
    return fminf(fmaxf(p, -16.0f), 16.0f);
}

// ============ 2D Sinusoidal Positional Encoding ============

__device__ __forceinline__ f32 sinusoidal_pe_2d(i32 py, i32 px, i32 dim, i32 out_dim) {
    const f32 base = 10000.0f;
    const i32 half_dim = out_dim / 2;
    
    if (dim < half_dim) {
        i32 i = dim / 2;
        f32 exponent = (2.0f * i) / (f32)half_dim;
        f32 divisor = powf(base, exponent);
        f32 angle = (f32)py / divisor;
        return (dim % 2 == 0) ? sinf(angle) : cosf(angle);
    } else {
        i32 local_dim = dim - half_dim;
        i32 i = local_dim / 2;
        f32 exponent = (2.0f * i) / (f32)half_dim;
        f32 divisor = powf(base, exponent);
        f32 angle = (f32)px / divisor;
        return (local_dim % 2 == 0) ? sinf(angle) : cosf(angle);
    }
}

// ============ Scoring Kernels ============

__global__ void compute_mse_scores(
    const f16* __restrict__ output,
    const f16* __restrict__ ground_truth,
          f32* __restrict__ scores,
    const i32 batch_size
) {
    const i32 batch_idx = blockIdx.x;
    const i32 tid = threadIdx.x;
    
    __shared__ f32 partial_sum[256];
    
    f32 local_sum = 0.0f;
    const i32 num_pixels = 64 * 64 * 3;
    
    for (i32 i = tid; i < num_pixels; i += blockDim.x) {
        f32 diff = __half2float(output[batch_idx * num_pixels + i]) - 
                   __half2float(ground_truth[i]);
        local_sum += diff * diff;
    }
    
    partial_sum[tid] = local_sum;
    __syncthreads();
    
    for (i32 stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        // Return negative MSE as fitness (higher = better)
        scores[batch_idx] = -partial_sum[0] / (f32)num_pixels;
    }
}

// Compute z-scored fitness values (EGGROLL style)
__global__ void compute_zscore_fitness(
    const f32* __restrict__ scores,
          f32* __restrict__ fitness_out,
    const i32 batch_size
) {
    const i32 tid = threadIdx.x;
    
    __shared__ f32 shared_scores[256];
    __shared__ f32 mean;
    __shared__ f32 variance;
    
    if (tid < batch_size) {
        shared_scores[tid] = scores[tid];
    }
    __syncthreads();
    
    // Compute mean
    if (tid == 0) {
        f32 sum = 0.0f;
        for (i32 i = 0; i < batch_size; i++) {
            sum += shared_scores[i];
        }
        mean = sum / (f32)batch_size;
    }
    __syncthreads();
    
    // Compute variance
    if (tid == 0) {
        f32 var_sum = 0.0f;
        for (i32 i = 0; i < batch_size; i++) {
            f32 diff = shared_scores[i] - mean;
            var_sum += diff * diff;
        }
        variance = var_sum / (f32)batch_size;
    }
    __syncthreads();
    
    // Z-score: (x - mean) / std
    if (tid < batch_size) {
        f32 std = sqrtf(variance + 1e-8f);
        fitness_out[tid] = (shared_scores[tid] - mean) / std;
    }
}

// ============ EGGROLL Accumulation Kernels ============
// Update: mu += (lr / N) * sum(E_i * f_i)
// Where E_i is rank-1 perturbation and f_i is z-scored fitness

__global__ void eggroll_accumulate_2d(
    const f16* __restrict__ base_weights,
          f16* __restrict__ out_weights,
    const f32* __restrict__ fitness,  // z-scored fitness values
    const i32 batch_size,
    const i32 M,
    const i32 N,
    const f32 sigma,
    const f32 lr,
    const u32 base_seed,
    const u32 layer_id
) {
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    const i32 total = M * N;
    
    if (idx >= total) return;
    
    const i32 row = idx / N;
    const i32 col = idx % N;
    
    f32 base_val = __half2float(base_weights[idx]);
    
    // EGGROLL update: (lr / N) * sum(E_i * f_i)
    f32 update = 0.0f;
    for (i32 b = 0; b < batch_size; b++) {
        f32 E_i = get_perturbation_eggroll(base_seed, b, layer_id, row, col, sigma);
        update += E_i * fitness[b];
    }
    update = (lr / (f32)batch_size) * update;
    
    out_weights[idx] = __float2half(base_val + update);
}

__global__ void eggroll_accumulate_3d(
    const f16* __restrict__ base_weights,
          f16* __restrict__ out_weights,
    const f32* __restrict__ fitness,
    const i32 batch_size,
    const i32 D1,
    const i32 D2,
    const i32 D3,
    const f32 sigma,
    const f32 lr,
    const u32 base_seed,
    const u32 layer_id
) {
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    const i32 total = D1 * D2 * D3;
    
    if (idx >= total) return;
    
    const i32 row = idx / D3;
    const i32 col = idx % D3;
    
    f32 base_val = __half2float(base_weights[idx]);
    
    f32 update = 0.0f;
    for (i32 b = 0; b < batch_size; b++) {
        f32 E_i = get_perturbation_eggroll(base_seed, b, layer_id, row, col, sigma);
        update += E_i * fitness[b];
    }
    update = (lr / (f32)batch_size) * update;
    
    out_weights[idx] = __float2half(base_val + update);
}

__global__ void eggroll_accumulate_4d(
    const f16* __restrict__ base_weights,
          f16* __restrict__ out_weights,
    const f32* __restrict__ fitness,
    const i32 batch_size,
    const i32 D1,
    const i32 D2,
    const i32 D3,
    const i32 D4,
    const f32 sigma,
    const f32 lr,
    const u32 base_seed,
    const u32 layer_id
) {
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    const i32 total = D1 * D2 * D3 * D4;
    
    if (idx >= total) return;
    
    // For [64, 3, 8, 8]: D1=64, D2=3, D3=8, D4=8
    const i32 d1 = idx / (D2 * D3 * D4);           // patch index
    const i32 rem = idx % (D2 * D3 * D4);
    const i32 d2 = rem / (D3 * D4);                // channel index
    const i32 spatial = rem % (D3 * D4);           // y*8 + x
    
    f32 base_val = __half2float(base_weights[idx]);
    
    f32 update = 0.0f;
    for (i32 b = 0; b < batch_size; b++) {
        // Use layer_id + channel for independent perturbations per channel
        f32 E_i = get_perturbation_eggroll(base_seed, b, layer_id + d2, d1, spatial, sigma);
        update += E_i * fitness[b];
    }
    update = (lr / (f32)batch_size) * update;
    
    out_weights[idx] = __float2half(base_val + update);
}

__global__ void eggroll_accumulate_5d(
    const f16* __restrict__ base_weights,
          f16* __restrict__ out_weights,
    const f32* __restrict__ fitness,
    const i32 batch_size,
    const i32 D1,
    const i32 D2,
    const i32 D3,
    const i32 D4,
    const i32 D5,
    const f32 sigma,
    const f32 lr,
    const u32 base_seed,
    const u32 layer_id
) {
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    const i32 total = D1 * D2 * D3 * D4 * D5;
    
    if (idx >= total) return;
    
    const i32 row = idx / D5;
    const i32 col = idx % D5;
    
    f32 base_val = __half2float(base_weights[idx]);
    
    f32 update = 0.0f;
    for (i32 b = 0; b < batch_size; b++) {
        f32 E_i = get_perturbation_eggroll(base_seed, b, layer_id, row, col, sigma);
        update += E_i * fitness[b];
    }
    update = (lr / (f32)batch_size) * update;
    
    out_weights[idx] = __float2half(base_val + update);
}

// ============ Patchifier with Positional Encoding ============

__global__ void patchify_and_project(
    const f16* __restrict__ images,
    const f16* __restrict__ proj_weights,
          f16* __restrict__ tokens,
    const i32 batch_size,
    const i32 out_dim,
    const f32 sigma,
    const u32 base_seed,
    const u32 layer_id,
    const f32 pe_scale
) {
    const i32 patch_group = blockIdx.x;
    const i32 batch_idx   = blockIdx.y;
    const i32 tid         = threadIdx.x;
    
    const i32 patch_start = patch_group * 16;
    const i32 num_out_chunks = out_dim / 16;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, f16, wmma::row_major> patch_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, f16, wmma::row_major> weight_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, f32> acc_frag;
    
    __shared__ f16 patch_smem[256];
    __shared__ f16 weight_smem[256];
    __shared__ f32 tmp_f32[256];
    __shared__ f32 accumulated[256];
    
    for (i32 out_chunk = 0; out_chunk < num_out_chunks; out_chunk++) {
        for (i32 i = tid; i < 256; i += 32) {
            accumulated[i] = 0.0f;
        }
        __syncthreads();
        
        for (i32 in_chunk = 0; in_chunk < 12; in_chunk++) {
            for (i32 i = tid; i < 256; i += 32) {
                const i32 local_patch = i / 16;
                const i32 local_dim = i % 16;
                
                const i32 patch_idx = patch_start + local_patch;
                const i32 py = patch_idx / 8;
                const i32 px = patch_idx % 8;
                
                const i32 flat_idx = in_chunk * 16 + local_dim;
                const i32 local_y = flat_idx / 24;
                const i32 local_x = (flat_idx % 24) / 3;
                const i32 channel = flat_idx % 3;
                
                const i32 img_y = py * 8 + local_y;
                const i32 img_x = px * 8 + local_x;
                
                f16 pixel = (flat_idx < 192) ? 
                    images[batch_idx * 64 * 64 * 3 + img_y * 64 * 3 + img_x * 3 + channel] : 
                    __float2half(0.0f);
                patch_smem[i] = pixel;
            }
            
            for (i32 i = tid; i < 256; i += 32) {
                const i32 row = i / 16;
                const i32 col = i % 16;
                const i32 weight_row = in_chunk * 16 + row;
                const i32 weight_col = out_chunk * 16 + col;
                
                f32 w = (weight_row < 192) ? 
                    __half2float(proj_weights[weight_row * out_dim + weight_col]) : 0.0f;
                w += get_perturbation_eggroll(base_seed, batch_idx, layer_id, weight_row, weight_col, sigma);
                weight_smem[i] = __float2half(w);
            }
            __syncthreads();
            
            wmma::load_matrix_sync(patch_frag, patch_smem, 16);
            wmma::load_matrix_sync(weight_frag, weight_smem, 16);
            wmma::fill_fragment(acc_frag, 0.0f);
            wmma::mma_sync(acc_frag, patch_frag, weight_frag, acc_frag);
            wmma::store_matrix_sync(tmp_f32, acc_frag, 16, wmma::mem_row_major);
            __syncthreads();
            
            for (i32 i = tid; i < 256; i += 32) {
                accumulated[i] += tmp_f32[i];
            }
            __syncthreads();
        }
        
        const i32 token_offset = batch_idx * 64 * out_dim + patch_start * out_dim + out_chunk * 16;
        for (i32 i = tid; i < 256; i += 32) {
            const i32 local_patch = i / 16;
            const i32 local_dim = i % 16;
            const i32 patch_idx = patch_start + local_patch;
            const i32 py = patch_idx / 8;
            const i32 px = patch_idx % 8;
            const i32 dim = out_chunk * 16 + local_dim;
            
            f32 pe = sinusoidal_pe_2d(py, px, dim, out_dim) * pe_scale;
            f32 val_with_pe = accumulated[i] + pe;
            
            tokens[token_offset + local_patch * out_dim + local_dim] = __float2half(val_with_pe);
        }
        __syncthreads();
    }
}

// ============ Final Projection ============

__global__ void final_projection(
    const f16* __restrict__ tokens,
    const f16* __restrict__ proj_weights,
          f16* __restrict__ output,
    const i32 batch_size,
    const f32 sigma,
    const u32 base_seed,
    const u32 layer_id
) {
    const i32 token_group = blockIdx.x;
    const i32 batch_idx   = blockIdx.y;
    const i32 tid         = threadIdx.x;
    
    const i32 token_start = token_group * 16;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, f16, wmma::row_major> tok_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, f16, wmma::row_major> weight_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, f32> acc_frag;
    
    __shared__ f16 tok_smem[256];
    __shared__ f16 weight_smem[256];
    __shared__ f32 tmp_f32[256];
    
    const i32 token_offset = batch_idx * 64 * 16 + token_start * 16;
    for (i32 i = tid; i < 256; i += 32) {
        tok_smem[i] = tokens[token_offset + i];
    }
    __syncthreads();
    
    wmma::load_matrix_sync(tok_frag, tok_smem, 16);
    
    for (i32 out_chunk = 0; out_chunk < 4; out_chunk++) {
        const i32 row_base = out_chunk * 16;
        
        for (i32 i = tid; i < 256; i += 32) {
            const i32 row = i / 16;
            const i32 col = i % 16;
            
            f32 w = __half2float(proj_weights[out_chunk * 256 + i]);
            w += get_perturbation_eggroll(base_seed, batch_idx, layer_id, row_base + row, col, sigma);
            weight_smem[i] = __float2half(w);
        }
        __syncthreads();
        
        wmma::load_matrix_sync(weight_frag, weight_smem, 16);
        wmma::fill_fragment(acc_frag, 0.0f);
        wmma::mma_sync(acc_frag, tok_frag, weight_frag, acc_frag);
        wmma::store_matrix_sync(tmp_f32, acc_frag, 16, wmma::mem_row_major);
        __syncthreads();
        
        const i32 out_offset = batch_idx * 64 * 64 + token_start * 64 + out_chunk * 16;
        for (i32 i = tid; i < 256; i += 32) {
            const i32 local_tok = i / 16;
            const i32 local_dim = i % 16;
            f32 val = fminf(fmaxf(tmp_f32[i], -16.0f), 16.0f);
            f32 tanh_val = tanhf(val);
            // f32 tanh_val = val;
            output[out_offset + local_tok * 64 + local_dim] = __float2half(tanh_val);
        }
        __syncthreads();
    }
}

// ============ Materialize Image ============

__global__ void materialize_image(
    const f16* __restrict__ patch_weights,
    const f16* __restrict__ learnable_patches,
          f16* __restrict__ output,
    const i32 batch_size,
    const f32 sigma,
    const u32 base_seed,
    const u32 layer_id
) {
    const i32 pos_idx   = blockIdx.x;
    const i32 batch_idx = blockIdx.y;
    const i32 tid       = threadIdx.x;
    
    const i32 py = pos_idx / 8;
    const i32 px = pos_idx % 8;
    
    __shared__ f32 weights[64];
    for (i32 i = tid; i < 64; i += 32) {
        weights[i] = __half2float(patch_weights[batch_idx * 64 * 64 + pos_idx * 64 + i]);
        // weights[i] = i == pos_idx ? 1.0f : 0.0f;
    }
    __syncthreads();
    
    for (i32 spatial = tid; spatial < 64; spatial += 32) {
        const i32 local_y = spatial / 8;
        const i32 local_x = spatial % 8;
        
        for (i32 channel = 0; channel < 3; channel++) {
            f32 blended = 0.0f;
            for (i32 p = 0; p < 64; p++) {
                f32 patch_pixel = __half2float(
                    learnable_patches[p * 192 + channel * 64 + local_y * 8 + local_x]
                );
                patch_pixel += get_perturbation_eggroll(base_seed, batch_idx, layer_id + channel, p, spatial, sigma);
                blended += patch_pixel * weights[p];
            }
            
            const i32 img_y = py * 8 + local_y;
            const i32 img_x = px * 8 + local_x;
            output[batch_idx * 64 * 64 * 3 + img_y * 64 * 3 + img_x * 3 + channel] = __float2half(blended);
        }
    }
}

// ============ QKV Projection ============

__global__ void compute_qkv_batched(
    const f16* __restrict__ tokens,
    const f16* __restrict__ QKV_weights,
          f16* __restrict__ Q_output,
          f16* __restrict__ K_output,
          f16* __restrict__ V_output,
    const i32 batch_size,
    const i32 num_tokens,
    const i32 num_heads,
    const f32 sigma,
    const u32 base_seed,
    const u32 layer_id
) {
    const i32 token_block = blockIdx.x;
    const i32 head_idx    = blockIdx.y;
    const i32 batch_idx   = blockIdx.z;
    const i32 tid         = threadIdx.x;
    
    const i32 in_dim = num_heads * 16;
    const i32 tokens_offset = batch_idx * num_tokens * in_dim + token_block * 16 * in_dim;
    const i32 output_offset = batch_idx * num_heads * num_tokens * 16 
                            + head_idx * num_tokens * 16 
                            + token_block * 256;
    
    const i32 weight_base = head_idx * 3 * num_heads * 256;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, f16, wmma::row_major> tok_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, f16, wmma::row_major> weight_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, f32> acc_frag;
    
    __shared__ f16 tok_smem[256];
    __shared__ f16 weight_smem[256];
    __shared__ f32 tmp_f32[256];
    __shared__ f32 acc_Q[256];
    __shared__ f32 acc_K[256];
    __shared__ f32 acc_V[256];
    
    for (i32 i = tid; i < 256; i += 32) {
        acc_Q[i] = 0.0f;
        acc_K[i] = 0.0f;
        acc_V[i] = 0.0f;
    }
    __syncthreads();
    
    for (i32 in_head = 0; in_head < num_heads; in_head++) {
        for (i32 i = tid; i < 256; i += 32) {
            const i32 local_tok = i / 16;
            const i32 local_dim = i % 16;
            tok_smem[i] = tokens[tokens_offset + local_tok * in_dim + in_head * 16 + local_dim];
        }
        __syncthreads();
        
        wmma::load_matrix_sync(tok_frag, tok_smem, 16);
        
        // Q
        const i32 q_weight_offset = weight_base + in_head * 256;
        const i32 q_row_base = ((head_idx * 3 + 0) * num_heads + in_head) * 16;
        for (i32 i = tid; i < 256; i += 32) {
            const i32 row = i / 16;
            const i32 col = i % 16;
            f32 w = __half2float(QKV_weights[q_weight_offset + i]);
            w += get_perturbation_eggroll(base_seed, batch_idx, layer_id, q_row_base + row, col, sigma);
            weight_smem[i] = __float2half(w);
        }
        __syncthreads();
        
        wmma::load_matrix_sync(weight_frag, weight_smem, 16);
        wmma::fill_fragment(acc_frag, 0.0f);
        wmma::mma_sync(acc_frag, tok_frag, weight_frag, acc_frag);
        wmma::store_matrix_sync(tmp_f32, acc_frag, 16, wmma::mem_row_major);
        __syncthreads();
        
        for (i32 i = tid; i < 256; i += 32) {
            acc_Q[i] += tmp_f32[i];
        }
        __syncthreads();
        
        // K
        const i32 k_weight_offset = weight_base + num_heads * 256 + in_head * 256;
        const i32 k_row_base = ((head_idx * 3 + 1) * num_heads + in_head) * 16;
        for (i32 i = tid; i < 256; i += 32) {
            const i32 row = i / 16;
            const i32 col = i % 16;
            f32 w = __half2float(QKV_weights[k_weight_offset + i]);
            w += get_perturbation_eggroll(base_seed, batch_idx, layer_id, k_row_base + row, col, sigma);
            weight_smem[i] = __float2half(w);
        }
        __syncthreads();
        
        wmma::load_matrix_sync(weight_frag, weight_smem, 16);
        wmma::fill_fragment(acc_frag, 0.0f);
        wmma::mma_sync(acc_frag, tok_frag, weight_frag, acc_frag);
        wmma::store_matrix_sync(tmp_f32, acc_frag, 16, wmma::mem_row_major);
        __syncthreads();
        
        for (i32 i = tid; i < 256; i += 32) {
            acc_K[i] += tmp_f32[i];
        }
        __syncthreads();
        
        // V
        const i32 v_weight_offset = weight_base + 2 * num_heads * 256 + in_head * 256;
        const i32 v_row_base = ((head_idx * 3 + 2) * num_heads + in_head) * 16;
        for (i32 i = tid; i < 256; i += 32) {
            const i32 row = i / 16;
            const i32 col = i % 16;
            f32 w = __half2float(QKV_weights[v_weight_offset + i]);
            w += get_perturbation_eggroll(base_seed, batch_idx, layer_id, v_row_base + row, col, sigma);
            weight_smem[i] = __float2half(w);
        }
        __syncthreads();
        
        wmma::load_matrix_sync(weight_frag, weight_smem, 16);
        wmma::fill_fragment(acc_frag, 0.0f);
        wmma::mma_sync(acc_frag, tok_frag, weight_frag, acc_frag);
        wmma::store_matrix_sync(tmp_f32, acc_frag, 16, wmma::mem_row_major);
        __syncthreads();
        
        for (i32 i = tid; i < 256; i += 32) {
            acc_V[i] += tmp_f32[i];
        }
        __syncthreads();
    }
    
    f32 l2_norm_Q = 0.0f;
    f32 l2_norm_K = 0.0f;
    f32 l2_norm_V = 0.0f;
    for (i32 i = tid; i < 256; i += 32) {
        l2_norm_Q += acc_Q[i] * acc_Q[i];
        l2_norm_K += acc_K[i] * acc_K[i];
        l2_norm_V += acc_V[i] * acc_V[i];
    }
    __syncthreads();

    for (i32 i = tid; i < 256; i += 32) {
        Q_output[output_offset + i] = __float2half(acc_Q[i] / sqrtf(l2_norm_Q + 1e-8f));  // normalize Q
        K_output[output_offset + i] = __float2half(acc_K[i] / sqrtf(l2_norm_K + 1e-8f));  // normalize K
        V_output[output_offset + i] = __float2half(acc_V[i] / sqrtf(l2_norm_V + 1e-8f));  // normalize V
    }
}

// ============ Flash Attention ============

__global__ void flash_attention_batched(
    const f16* __restrict__ Q,
    const f16* __restrict__ K,
    const f16* __restrict__ V,
          f16* __restrict__ O,
    const i32 batch_size,
    const i32 num_heads,
    const i32 num_q,
    const i32 num_kv
) {
    const i32 q_block   = blockIdx.x;
    const i32 head_idx  = blockIdx.y;
    const i32 batch_idx = blockIdx.z;
    const i32 tid       = threadIdx.x;
    
    const i32 qo_offset = batch_idx * num_heads * num_q * 16 
                        + head_idx * num_q * 16 
                        + q_block * 256;
    const i32 kv_base   = batch_idx * num_heads * num_kv * 16 
                        + head_idx * num_kv * 16;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, f16, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, f16, wmma::row_major> k_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, f32> qk_acc_frag;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, f16, wmma::row_major> attn_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, f16, wmma::row_major> v_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, f32> out_frag;
    
    __shared__ f32 row_max[16];
    __shared__ f32 row_sum[16];
    __shared__ f16 Q_smem[256];
    __shared__ f16 K_smem[256];
    __shared__ f16 V_smem[256];
    __shared__ f16 attn_smem[256];
    __shared__ f32 scores[256];
    __shared__ f32 output[256];
    __shared__ f32 tmp_f32[256];
    
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
        wmma::fill_fragment(qk_acc_frag, 0.0f);
        wmma::mma_sync(qk_acc_frag, q_frag, k_frag, qk_acc_frag);
        wmma::store_matrix_sync(tmp_f32, qk_acc_frag, 16, wmma::mem_row_major);
        __syncthreads();
        
        for (i32 i = tid; i < 256; i += 32) {
            scores[i] = tmp_f32[i] * 16.0f; // Temperature scaling
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
            i32 dim   = i % 16;
            attn_smem[i] = __float2half(scores[i]);
            V_smem[i] = V[kv_offset + token * 16 + dim];
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
        // O[qo_offset + i] = __float2half(max(output[i], 0.0f));
        O[qo_offset + i] = __float2half(tanhf(output[i]));
    }
}

// ============ Project and Residual ============

__global__ void project_and_residual(
    const f16* __restrict__ attn_output,
    const f16* __restrict__ proj_weights,
    const f16* __restrict__ residual,
          f16* __restrict__ output,
    const i32 batch_size,
    const i32 num_heads,
    const i32 num_tokens,
    const f32 sigma,
    const u32 base_seed,
    const u32 layer_id
) {
    const i32 token_block = blockIdx.x;
    const i32 batch_idx   = blockIdx.y;
    const i32 tid         = threadIdx.x;
    
    const i32 out_dim = num_heads * 16;
    const i32 residual_offset = batch_idx * num_tokens * out_dim + token_block * 16 * out_dim;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, f16, wmma::row_major> attn_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, f16, wmma::row_major> proj_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, f32> acc_frag;
    
    __shared__ f16 attn_smem[256];
    __shared__ f16 proj_smem[256];
    __shared__ f32 tmp_f32[256];
    __shared__ f32 accumulated[256];
    
    for (i32 head_idx = 0; head_idx < num_heads; head_idx++) {
        for (i32 i = tid; i < 256; i += 32) {
            const i32 local_tok = i / 16;
            const i32 local_dim = i % 16;
            accumulated[i] = __half2float(
                residual[residual_offset + local_tok * out_dim + head_idx * 16 + local_dim]
            );
        }
        __syncthreads();
        
        const i32 attn_offset = batch_idx * num_heads * num_tokens * 16 
                              + head_idx * num_tokens * 16 
                              + token_block * 256;
        const i32 proj_offset = head_idx * 256;
        const i32 proj_row_base = head_idx * 16;
        
        for (i32 i = tid; i < 256; i += 32) {
            attn_smem[i] = attn_output[attn_offset + i];
            
            const i32 row = i / 16;
            const i32 col = i % 16;
            f32 w = __half2float(proj_weights[proj_offset + i]);
            w += get_perturbation_eggroll(base_seed, batch_idx, layer_id, proj_row_base + row, col, sigma);
            proj_smem[i] = __float2half(w);
        }
        __syncthreads();
        
        wmma::load_matrix_sync(attn_frag, attn_smem, 16);
        wmma::load_matrix_sync(proj_frag, proj_smem, 16);
        wmma::fill_fragment(acc_frag, 0.0f);
        wmma::mma_sync(acc_frag, attn_frag, proj_frag, acc_frag);
        wmma::store_matrix_sync(tmp_f32, acc_frag, 16, wmma::mem_row_major);
        __syncthreads();
        
        for (i32 i = tid; i < 256; i += 32) {
            accumulated[i] += tmp_f32[i];
        }
        __syncthreads();
        
        for (i32 i = tid; i < 256; i += 32) {
            const i32 local_tok = i / 16;
            const i32 local_dim = i % 16;
            output[residual_offset + local_tok * out_dim + head_idx * 16 + local_dim] = 
                __float2half(accumulated[i]);
        }
        __syncthreads();
    }
}

// ============ Host Functions ============

torch::Tensor patchify_fn(
    int64_t stream_ptr,
    torch::Tensor images,
    torch::Tensor proj_weights,
    double sigma,
    int64_t seed,
    int64_t layer_id,
    double pe_scale
) {
    const i32 batch_size = images.size(0);
    const i32 out_dim = proj_weights.size(1);
    
    auto tokens = torch::empty({batch_size, 64, out_dim}, 
                               torch::dtype(torch::kHalf).device(images.device()));
    
    dim3 grid(4, batch_size);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    patchify_and_project<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const f16*>(images.data_ptr<at::Half>()),
        reinterpret_cast<const f16*>(proj_weights.data_ptr<at::Half>()),
        reinterpret_cast<f16*>(tokens.data_ptr<at::Half>()),
        batch_size,
        out_dim,
        static_cast<f32>(sigma),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id),
        static_cast<f32>(pe_scale)
    );
    
    return tokens;
}

torch::Tensor final_projection_fn(
    int64_t stream_ptr,
    torch::Tensor tokens,
    torch::Tensor proj_weights,
    double sigma,
    int64_t seed,
    int64_t layer_id
) {
    const i32 batch_size = tokens.size(0);
    
    auto output = torch::empty({batch_size, 64, 64}, 
                               torch::dtype(torch::kHalf).device(tokens.device()));
    
    dim3 grid(4, batch_size);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    final_projection<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const f16*>(tokens.data_ptr<at::Half>()),
        reinterpret_cast<const f16*>(proj_weights.data_ptr<at::Half>()),
        reinterpret_cast<f16*>(output.data_ptr<at::Half>()),
        batch_size,
        static_cast<f32>(sigma),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
    );
    
    return output;
}

torch::Tensor materialize_image_fn(
    int64_t stream_ptr,
    torch::Tensor patch_weights,
    torch::Tensor learnable_patches,
    double sigma,
    int64_t seed,
    int64_t layer_id
) {
    const i32 batch_size = patch_weights.size(0);
    
    auto output = torch::empty({batch_size, 64, 64, 3}, 
                               torch::dtype(torch::kHalf).device(patch_weights.device()));
    
    dim3 grid(64, batch_size);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    materialize_image<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const f16*>(patch_weights.data_ptr<at::Half>()),
        reinterpret_cast<const f16*>(learnable_patches.data_ptr<at::Half>()),
        reinterpret_cast<f16*>(output.data_ptr<at::Half>()),
        batch_size,
        static_cast<f32>(sigma),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
    );
    
    return output;
}

std::vector<torch::Tensor> compute_qkv_batched_fn(
    int64_t stream_ptr,
    torch::Tensor tokens,
    torch::Tensor QKV_weights,
    double sigma,
    int64_t seed,
    int64_t layer_id
) {
    const i32 batch_size = tokens.size(0);
    const i32 num_tokens = tokens.size(1);
    const i32 num_heads  = QKV_weights.size(0);
    
    auto Q_out = torch::empty({batch_size, num_heads, num_tokens, 16}, 
                              torch::dtype(torch::kHalf).device(tokens.device()));
    auto K_out = torch::empty({batch_size, num_heads, num_tokens, 16}, 
                              torch::dtype(torch::kHalf).device(tokens.device()));
    auto V_out = torch::empty({batch_size, num_heads, num_tokens, 16}, 
                              torch::dtype(torch::kHalf).device(tokens.device()));
    
    dim3 grid(num_tokens / 16, num_heads, batch_size);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    compute_qkv_batched<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const f16*>(tokens.data_ptr<at::Half>()),
        reinterpret_cast<const f16*>(QKV_weights.data_ptr<at::Half>()),
        reinterpret_cast<f16*>(Q_out.data_ptr<at::Half>()),
        reinterpret_cast<f16*>(K_out.data_ptr<at::Half>()),
        reinterpret_cast<f16*>(V_out.data_ptr<at::Half>()),
        batch_size,
        num_tokens,
        num_heads,
        static_cast<f32>(sigma),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
    );
    
    return {Q_out, K_out, V_out};
}

torch::Tensor flash_attention_batched_fn(
    int64_t stream_ptr,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    const i32 batch_size = Q.size(0);
    const i32 num_heads  = Q.size(1);
    const i32 num_q      = Q.size(2);
    const i32 num_kv     = K.size(2);
    
    auto O = torch::empty({batch_size, num_heads, num_q, 16}, 
                          torch::dtype(torch::kHalf).device(Q.device()));
    
    dim3 grid(num_q / 16, num_heads, batch_size);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    flash_attention_batched<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const f16*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const f16*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const f16*>(V.data_ptr<at::Half>()),
        reinterpret_cast<f16*>(O.data_ptr<at::Half>()),
        batch_size,
        num_heads,
        num_q,
        num_kv
    );
    
    return O;
}

torch::Tensor project_and_residual_fn(
    int64_t stream_ptr,
    torch::Tensor attn_output,
    torch::Tensor proj_weights,
    torch::Tensor residual,
    double sigma,
    int64_t seed,
    int64_t layer_id
) {
    const i32 batch_size = attn_output.size(0);
    const i32 num_heads  = attn_output.size(1);
    const i32 num_tokens = attn_output.size(2);
    
    auto output = torch::empty({batch_size, num_tokens, num_heads * 16}, 
                               torch::dtype(torch::kHalf).device(attn_output.device()));
    
    dim3 grid(num_tokens / 16, batch_size);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    project_and_residual<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const f16*>(attn_output.data_ptr<at::Half>()),
        reinterpret_cast<const f16*>(proj_weights.data_ptr<at::Half>()),
        reinterpret_cast<const f16*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<f16*>(output.data_ptr<at::Half>()),
        batch_size,
        num_heads,
        num_tokens,
        static_cast<f32>(sigma),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
    );
    
    return output;
}

torch::Tensor compute_mse_scores_fn(
    int64_t stream_ptr,
    torch::Tensor output,
    torch::Tensor ground_truth
) {
    const i32 batch_size = output.size(0);
    
    auto scores = torch::empty({batch_size}, 
                               torch::dtype(torch::kFloat32).device(output.device()));
    
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    compute_mse_scores<<<batch_size, 256, 0, stream>>>(
        reinterpret_cast<const f16*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const f16*>(ground_truth.data_ptr<at::Half>()),
        scores.data_ptr<f32>(),
        batch_size
    );
    
    return scores;
}

torch::Tensor compute_zscore_fitness_fn(
    int64_t stream_ptr,
    torch::Tensor scores
) {
    const i32 batch_size = scores.size(0);
    
    auto fitness = torch::empty({batch_size}, 
                                torch::dtype(torch::kFloat32).device(scores.device()));
    
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    compute_zscore_fitness<<<1, 256, 0, stream>>>(
        scores.data_ptr<f32>(),
        fitness.data_ptr<f32>(),
        batch_size
    );
    
    return fitness;
}

torch::Tensor eggroll_accumulate_2d_fn(
    int64_t stream_ptr,
    torch::Tensor base_weights,
    torch::Tensor fitness,
    double sigma,
    double lr,
    int64_t seed,
    int64_t layer_id
) {
    const i32 batch_size = fitness.size(0);
    const i32 M = base_weights.size(0);
    const i32 N = base_weights.size(1);
    const i32 total = M * N;
    
    auto out_weights = torch::empty_like(base_weights);
    
    const i32 threads = 256;
    const i32 blocks = (total + threads - 1) / threads;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    eggroll_accumulate_2d<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const f16*>(base_weights.data_ptr<at::Half>()),
        reinterpret_cast<f16*>(out_weights.data_ptr<at::Half>()),
        fitness.data_ptr<f32>(),
        batch_size,
        M, N,
        static_cast<f32>(sigma),
        static_cast<f32>(lr),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
    );
    
    return out_weights;
}

torch::Tensor eggroll_accumulate_3d_fn(
    int64_t stream_ptr,
    torch::Tensor base_weights,
    torch::Tensor fitness,
    double sigma,
    double lr,
    int64_t seed,
    int64_t layer_id
) {
    const i32 batch_size = fitness.size(0);
    const i32 D1 = base_weights.size(0);
    const i32 D2 = base_weights.size(1);
    const i32 D3 = base_weights.size(2);
    const i32 total = D1 * D2 * D3;
    
    auto out_weights = torch::empty_like(base_weights);
    
    const i32 threads = 256;
    const i32 blocks = (total + threads - 1) / threads;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    eggroll_accumulate_3d<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const f16*>(base_weights.data_ptr<at::Half>()),
        reinterpret_cast<f16*>(out_weights.data_ptr<at::Half>()),
        fitness.data_ptr<f32>(),
        batch_size,
        D1, D2, D3,
        static_cast<f32>(sigma),
        static_cast<f32>(lr),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
    );
    
    return out_weights;
}

torch::Tensor eggroll_accumulate_4d_fn(
    int64_t stream_ptr,
    torch::Tensor base_weights,
    torch::Tensor fitness,
    double sigma,
    double lr,
    int64_t seed,
    int64_t layer_id
) {
    const i32 batch_size = fitness.size(0);
    const i32 D1 = base_weights.size(0);
    const i32 D2 = base_weights.size(1);
    const i32 D3 = base_weights.size(2);
    const i32 D4 = base_weights.size(3);
    const i32 total = D1 * D2 * D3 * D4;
    
    auto out_weights = torch::empty_like(base_weights);
    
    const i32 threads = 256;
    const i32 blocks = (total + threads - 1) / threads;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    eggroll_accumulate_4d<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const f16*>(base_weights.data_ptr<at::Half>()),
        reinterpret_cast<f16*>(out_weights.data_ptr<at::Half>()),
        fitness.data_ptr<f32>(),
        batch_size,
        D1, D2, D3, D4,
        static_cast<f32>(sigma),
        static_cast<f32>(lr),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
    );
    
    return out_weights;
}

torch::Tensor eggroll_accumulate_5d_fn(
    int64_t stream_ptr,
    torch::Tensor base_weights,
    torch::Tensor fitness,
    double sigma,
    double lr,
    int64_t seed,
    int64_t layer_id
) {
    const i32 batch_size = fitness.size(0);
    const i32 D1 = base_weights.size(0);
    const i32 D2 = base_weights.size(1);
    const i32 D3 = base_weights.size(2);
    const i32 D4 = base_weights.size(3);
    const i32 D5 = base_weights.size(4);
    const i32 total = D1 * D2 * D3 * D4 * D5;
    
    auto out_weights = torch::empty_like(base_weights);
    
    const i32 threads = 256;
    const i32 blocks = (total + threads - 1) / threads;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    eggroll_accumulate_5d<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const f16*>(base_weights.data_ptr<at::Half>()),
        reinterpret_cast<f16*>(out_weights.data_ptr<at::Half>()),
        fitness.data_ptr<f32>(),
        batch_size,
        D1, D2, D3, D4, D5,
        static_cast<f32>(sigma),
        static_cast<f32>(lr),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
    );
    
    return out_weights;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("patchify_fn", &patchify_fn);
    m.def("final_projection_fn", &final_projection_fn);
    m.def("materialize_image_fn", &materialize_image_fn);
    m.def("compute_qkv_batched_fn", &compute_qkv_batched_fn);
    m.def("flash_attention_batched_fn", &flash_attention_batched_fn);
    m.def("project_and_residual_fn", &project_and_residual_fn);
    m.def("compute_mse_scores_fn", &compute_mse_scores_fn);
    m.def("compute_zscore_fitness_fn", &compute_zscore_fitness_fn);
    m.def("eggroll_accumulate_2d_fn", &eggroll_accumulate_2d_fn);
    m.def("eggroll_accumulate_3d_fn", &eggroll_accumulate_3d_fn);
    m.def("eggroll_accumulate_4d_fn", &eggroll_accumulate_4d_fn);
    m.def("eggroll_accumulate_5d_fn", &eggroll_accumulate_5d_fn);
}
;//
"""

cpp_source = """
torch::Tensor patchify_fn(int64_t stream_ptr, torch::Tensor images, torch::Tensor proj_weights, double sigma, int64_t seed, int64_t layer_id, double pe_scale);
torch::Tensor final_projection_fn(int64_t stream_ptr, torch::Tensor tokens, torch::Tensor proj_weights, double sigma, int64_t seed, int64_t layer_id);
torch::Tensor materialize_image_fn(int64_t stream_ptr, torch::Tensor patch_weights, torch::Tensor learnable_patches, double sigma, int64_t seed, int64_t layer_id);
std::vector<torch::Tensor> compute_qkv_batched_fn(int64_t stream_ptr, torch::Tensor tokens, torch::Tensor QKV_weights, double sigma, int64_t seed, int64_t layer_id);
torch::Tensor flash_attention_batched_fn(int64_t stream_ptr, torch::Tensor Q, torch::Tensor K, torch::Tensor V);
torch::Tensor project_and_residual_fn(int64_t stream_ptr, torch::Tensor attn_output, torch::Tensor proj_weights, torch::Tensor residual, double sigma, int64_t seed, int64_t layer_id);
torch::Tensor compute_mse_scores_fn(int64_t stream_ptr, torch::Tensor output, torch::Tensor ground_truth);
torch::Tensor compute_zscore_fitness_fn(int64_t stream_ptr, torch::Tensor scores);
torch::Tensor eggroll_accumulate_2d_fn(int64_t stream_ptr, torch::Tensor base_weights, torch::Tensor fitness, double sigma, double lr, int64_t seed, int64_t layer_id);
torch::Tensor eggroll_accumulate_3d_fn(int64_t stream_ptr, torch::Tensor base_weights, torch::Tensor fitness, double sigma, double lr, int64_t seed, int64_t layer_id);
torch::Tensor eggroll_accumulate_4d_fn(int64_t stream_ptr, torch::Tensor base_weights, torch::Tensor fitness, double sigma, double lr, int64_t seed, int64_t layer_id);
torch::Tensor eggroll_accumulate_5d_fn(int64_t stream_ptr, torch::Tensor base_weights, torch::Tensor fitness, double sigma, double lr, int64_t seed, int64_t layer_id);
"""

print("Compiling EGGROLL CUDA kernels...")
module = torch.utils.cpp_extension.load_inline(
    name="vit_eggroll",
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    extra_cuda_cflags=["-O3", "-arch=sm_80"],
    verbose=False
)
print("Compilation complete!")


# ============ Model Weights Container ============

class ViTWeights:
    def __init__(self, num_heads, num_layers, device="cuda"):
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device
        
        out_dim = num_heads * 16
        
        # Patchifier: [192, out_dim]
        self.patchifier = (torch.randn(192, out_dim, device=device) / math.sqrt(192)).half()
        
        # Final projection: [4, 16, 16]
        self.final_proj = (torch.randn(4, 16, 16, device=device) / math.sqrt(16)).half()
        
        # Learnable patches: [64, 3, 8, 8]
        self.learnable_patches = (torch.randn(64, 3, 8, 8, device=device) * 0.3).half()
        
        # Per-layer weights
        self.qkv_weights = []
        self.proj_weights = []
        for _ in range(num_layers):
            qkv = (torch.randn(num_heads, 3, num_heads, 16, 16, device=device) / math.sqrt(16 * num_heads)).half()
            self.qkv_weights.append(qkv)
            
            proj = (torch.randn(num_heads, 16, 16, device=device) / math.sqrt(16 * num_heads)).half()
            self.proj_weights.append(proj)
    
    def accumulate_perturbations(self, fitness, seed, sigma, lr, stream):
        """
        EGGROLL update: mu += (lr / N) * sum(E_i * f_i)
        
        fitness: z-scored fitness values [batch_size]
        sigma: perturbation scale
        lr: learning rate
        """
        # Patchifier
        self.patchifier = module.eggroll_accumulate_2d_fn(
            stream.cuda_stream, self.patchifier, fitness, sigma, lr, seed, 0
        )
        
        # Final projection
        self.final_proj = module.eggroll_accumulate_3d_fn(
            stream.cuda_stream, self.final_proj, fitness, sigma, lr, seed, 1
        )
        
        # Learnable patches
        self.learnable_patches = module.eggroll_accumulate_4d_fn(
            stream.cuda_stream, self.learnable_patches, fitness, sigma, lr, seed, 2
        )
        
        # Per-layer weights
        for i in range(self.num_layers):
            layer_base = 100 + i * 10
            
            self.qkv_weights[i] = module.eggroll_accumulate_5d_fn(
                stream.cuda_stream, self.qkv_weights[i], fitness, sigma, lr, seed, layer_base
            )
            
            self.proj_weights[i] = module.eggroll_accumulate_3d_fn(
                stream.cuda_stream, self.proj_weights[i], fitness, sigma, lr, seed, layer_base + 1
            )
    
        stream.synchronize()
        # Apply L2 decay

        decay = 0.9999
        self.patchifier *= decay
        self.final_proj *= decay
        self.learnable_patches *= decay
        for i in range(self.num_layers):
            self.qkv_weights[i] *= decay
            self.proj_weights[i] *= decay


# ============ Forward Pass ============

def forward_pass(images, weights, seed, sigma, pe_scale, stream):
    """Forward pass with EGGROLL perturbations"""
    
    # 1. Patchify
    tokens = module.patchify_fn(
        stream.cuda_stream, images, weights.patchifier, sigma, seed, 0, pe_scale
    )
    
    # 2. Attention layers
    x = tokens
    for i, (qkv_w, proj_w) in enumerate(zip(weights.qkv_weights, weights.proj_weights)):
        layer_base = 100 + i * 10
        
        Q, K, V = module.compute_qkv_batched_fn(
            stream.cuda_stream, x, qkv_w, sigma, seed, layer_base
        )
        
        attn_out = module.flash_attention_batched_fn(stream.cuda_stream, Q, K, V)
        
        x = module.project_and_residual_fn(
            stream.cuda_stream, attn_out, proj_w, x, sigma, seed, layer_base + 1
        )
    
    # 3. Reduce to [batch, 64, 16]
    x_reduced = x[:, :, :16].contiguous()
    
    # 4. Final projection
    patch_weights = module.final_projection_fn(
        stream.cuda_stream, x_reduced, weights.final_proj, sigma, seed, 1
    )
    
    # 5. Materialize image
    output = module.materialize_image_fn(
        stream.cuda_stream, patch_weights, weights.learnable_patches, sigma, seed, 2
    )
    
    return output


# ============ Training ============

def train_step(images, ground_truth, weights, seed, sigma, lr, pe_scale, stream):
    """
    EGGROLL training step:
    1. Forward pass with perturbations
    2. Compute MSE scores
    3. Z-score fitness normalization
    4. Update weights: mu += (lr / N) * sum(E_i * f_i)
    """
    # Forward pass
    output = forward_pass(images, weights, seed, sigma, pe_scale, stream)
    
    # Compute scores (negative MSE = fitness, higher is better)
    scores = module.compute_mse_scores_fn(stream.cuda_stream, output, ground_truth)
    
    # Z-score normalization (EGGROLL style)
    fitness = module.compute_zscore_fitness_fn(stream.cuda_stream, scores)
    
    # Update weights
    weights.accumulate_perturbations(fitness, seed, sigma, lr, stream)
    
    return output, scores, fitness


# ============ Data Loading ============

class SimpleDataloader:
    def __init__(self, path):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".png")]
    
    def get_image(self):
        img_path = random.choice(self.paths)
        return self.transform(Image.open(img_path))


# ============ Main ============

if __name__ == "__main__":
    print("=" * 60)
    print("EGGROLL ViT Training (FP16)")
    print("Based on: Evolution Strategies at the Hyperscale")
    print("=" * 60)
    
    # Config - EGGROLL style
    batch_size      = 256    # Population size N
    num_heads       = 8
    num_layers      = 8
    sigma           = 0.1    # Perturbation scale (sigma in paper)
    lr              = 1.0e-4    # Learning rate (alpha in paper)
    pe_scale        = 1.0
    num_epochs      = 1 << 14
    
    # Setup
    stream = torch.cuda.Stream()
    weights = ViTWeights(num_heads, num_layers)
    
    # Ground truth - simple gradient for testing
    device = "cuda"

    # Load image
    class SimpleDataloader:
        def __init__(self, path):
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            self.paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".png")]
        
        def get_image(self):
            img_path = random.choice(self.paths)
            return self.transform(Image.open(img_path))

    dataset = SimpleDataloader("data\\MNIST\\dataset-part1")
    ground_truth_f = dataset.get_image().to("cuda").permute(1, 2, 0).contiguous()
    ground_truth = ground_truth_f.unsqueeze(0).repeat(batch_size, 1, 1, 1).half()

    print(f"\nConfig:")
    print(f"  Population size (N): {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Num layers: {num_layers}")
    print(f"  Sigma: {sigma}")
    print(f"  Learning rate: {lr}")
    print(f"  PE scale: {pe_scale}")
    print(f"  Ground truth range: [{ground_truth.min():.2f}, {ground_truth.max():.2f}]")
    print()
    
    # Training loop
    losses = []
    images = ground_truth.clone()
    
    for epoch in range(num_epochs):
        seed = epoch * 1000
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record(stream)
        
        # Add noise to input
        noise_scale = 0.05
        noisy_images = images * (1.0 - noise_scale) + noise_scale * (torch.rand_like(images) * 2 - 1)
        
        output, scores, fitness = train_step(
            noisy_images, ground_truth, weights, seed, sigma, lr, pe_scale, stream
        )
        
        # Optional: decay sigma over time (paper doesn't require this but can help)
        sigma *= 0.9999
        
        end.record(stream)
        end.synchronize()
        
        elapsed_ms = start.elapsed_time(end)
        
        scores_np = scores.cpu().numpy()
        fitness_np = fitness.cpu().numpy()
        
        best_mse = -scores_np.max()
        mean_mse = -scores_np.mean()
        losses.append(best_mse)
        
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            try:
                from py.torch_utils import dds_from_tensor
                dds = dds_from_tensor(output[0:1].float().permute(0, 3, 1, 2))
                dds.save(".tmp/epoch.dds")
            except: pass
            
            print(f"Epoch {epoch + 1}:")
            print(f"  Time: {elapsed_ms:.2f} ms")
            print(f"  Best MSE: {best_mse:.6f}, Mean MSE: {mean_mse:.6f}")
            print(f"  Fitness range: [{fitness_np.min():.2f}, {fitness_np.max():.2f}]")
            print(f"  Fitness std: {fitness_np.std():.4f}")
            print(f"  Output range: [{output.min():.2f}, {output.max():.2f}]")
            print()
    
    print("=" * 60)
    print(f"Loss progression: {losses[0]:.6f} -> {losses[-1]:.6f}")
    if losses[0] > 0:
        print(f"Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    print("=" * 60)
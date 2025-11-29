# MIT License
# Copyright (c) 2025 Anton Schreiner

import math
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline
import torch.nn as nn
import os
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import random

from unittest import result
from py.torch_utils import dds_from_tensor, DelayedKeyboardInterrupt
from piq import SSIMLoss, LPIPS
import math
import numpy as np
from pytorch_optimizer import Muon, SOAP, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, PolynomialLR
import argparse
from einops import rearrange, reduce, repeat


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

// ============ Random Number Generation ============

static __device__ __forceinline__ u32 lowbias32(u32 x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

static __device__ __forceinline__ f32 random_normal(u32 seed) {
    u32 s1 = lowbias32(seed);
    u32 s2 = lowbias32(s1);
    
    f32 u1 = fmaxf((f32)(s1 & 0xFFFF) / 65535.0f, 1e-7f);
    f32 u2 = (f32)(s2 & 0xFFFF) / 65535.0f;
    
    f32 radius = sqrtf(-2.0f * logf(u1));
    f32 theta = 2.0f * 3.14159265358979323846f * u2;
    return radius * cosf(theta);
}

static __device__ __forceinline__ f32 get_perturbation(
    u32 base_seed, i32 batch_idx, u32 layer_id, i32 row, i32 col, f32 scale
) {
    u32 u_seed = base_seed ^ (batch_idx * 12345) ^ (layer_id * 111111) ^ (row * 67890);
    u32 v_seed = base_seed ^ (batch_idx * 12345) ^ (layer_id * 111111) ^ ((col + 100000) * 67890);
    
    f32 u_val = random_normal(u_seed) * scale;
    f32 v_val = random_normal(v_seed) * scale;
    
    return u_val * v_val;
}

__device__ __forceinline__ i8 quantize(f32 val, f32 inv_scale) {
    return static_cast<i8>(min(max(rintf(val * inv_scale), -127.0f), 127.0f));
}

// ============ 2D Sinusoidal Positional Encoding ============

static __device__ __forceinline__ f32 sinusoidal_pe_2d(i32 py, i32 px, i32 dim, i32 out_dim) {
    const f32 base = 10000.0f;
    const i32 half_dim = out_dim / 2;
    
    if (dim < half_dim) {
        i32 i = dim / 2;
        f32 exponent = (2.0f * i) / static_cast<f32>(half_dim);
        f32 divisor = powf(base, exponent);
        f32 angle = static_cast<f32>(py) / divisor;
        return (dim % 2 == 0) ? sinf(angle) : cosf(angle);
    } else {
        i32 local_dim = dim - half_dim;
        i32 i = local_dim / 2;
        f32 exponent = (2.0f * i) / static_cast<f32>(half_dim);
        f32 divisor = powf(base, exponent);
        f32 angle = static_cast<f32>(px) / divisor;
        return (local_dim % 2 == 0) ? sinf(angle) : cosf(angle);
    }
}

// ============ Scoring and Weight Update Kernels ============

__global__ void compute_mse_scores(
    const i8*  __restrict__ output,
    const i8*  __restrict__ ground_truth,
          f32* __restrict__ scores,
    const i32 batch_size
) {
    const i32 batch_idx = blockIdx.x;
    const i32 tid = threadIdx.x;
    
    __shared__ f32 partial_sum[256];
    
    f32 local_sum = 0.0f;
    const i32 num_pixels = 64 * 64 * 3;
    
    for (i32 i = tid; i < num_pixels; i += blockDim.x) {
        f32 diff = static_cast<f32>(output[batch_idx * num_pixels + i]) - 
                   static_cast<f32>(ground_truth[batch_idx * num_pixels + i]);
        local_sum += diff * diff  / (127.0f * 127.0f);
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
        scores[batch_idx] = -partial_sum[0] / static_cast<f32>(num_pixels);
    }
}

__global__ void compute_softmax(
    const f32* __restrict__ scores,
          f32* __restrict__ softmax_out,
    const i32 batch_size,
    const f32 temperature
) {
    const i32 tid = threadIdx.x;
    
    __shared__ f32 shared_scores[256];
    __shared__ f32 max_score;
    __shared__ f32 sum_exp;
    
    if (tid < batch_size) {
        shared_scores[tid] = scores[tid] / temperature;
    }
    __syncthreads();
    
    if (tid == 0) {
        max_score = shared_scores[0];
        for (i32 i = 1; i < batch_size; i++) {
            max_score = fmaxf(max_score, shared_scores[i]);
        }
    }
    __syncthreads();
    
    if (tid < batch_size) {
        shared_scores[tid] = expf(shared_scores[tid] - max_score);
    }
    __syncthreads();
    
    if (tid == 0) {
        sum_exp = 0.0f;
        for (i32 i = 0; i < batch_size; i++) {
            sum_exp += shared_scores[i];
        }
    }
    __syncthreads();
    
    if (tid < batch_size) {
        softmax_out[tid] = shared_scores[tid] / sum_exp;
    }
}

__global__ void accumulate_perturbation_2d(
    const i8*  __restrict__ base_weights,
          i8*  __restrict__ out_weights,
    const f32* __restrict__ softmax_weights,
    const i32 batch_size,
    const i32 M,
    const i32 N,
    const f32 scale,
    const f32 perturb_scale,
    const u32 base_seed,
    const u32 layer_id
) {
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    const i32 total = M * N;
    
    if (idx >= total) return;
    
    const i32 row = idx / N;
    const i32 col = idx % N;
    
    f32 base_val = static_cast<f32>(base_weights[idx]) * scale;
    
    f32 perturb_sum = 0.0f;
    for (i32 b = 0; b < batch_size; b++) {
        f32 p = get_perturbation(base_seed, b, layer_id, row, col, perturb_scale);
        perturb_sum += softmax_weights[b] * p;
    }
    
    f32 new_val = base_val + perturb_sum;
    out_weights[idx] = quantize(new_val, 1.0f / scale);
}

__global__ void accumulate_perturbation_3d(
    const i8*  __restrict__ base_weights,
          i8*  __restrict__ out_weights,
    const f32* __restrict__ softmax_weights,
    const i32 batch_size,
    const i32 D1,
    const i32 D2,
    const i32 D3,
    const f32 scale,
    const f32 perturb_scale,
    const u32 base_seed,
    const u32 layer_id
) {
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    const i32 total = D1 * D2 * D3;
    
    if (idx >= total) return;
    
    const i32 row = idx / D3;
    const i32 col = idx % D3;
    
    f32 base_val = static_cast<f32>(base_weights[idx]) * scale;
    
    f32 perturb_sum = 0.0f;
    for (i32 b = 0; b < batch_size; b++) {
        f32 p = get_perturbation(base_seed, b, layer_id, row, col, perturb_scale);
        perturb_sum += softmax_weights[b] * p;
    }
    
    f32 new_val = base_val + perturb_sum;
    out_weights[idx] = quantize(new_val, 1.0f / scale);
}

__global__ void accumulate_perturbation_5d(
    const i8*  __restrict__ base_weights,
          i8*  __restrict__ out_weights,
    const f32* __restrict__ softmax_weights,
    const i32 batch_size,
    const i32 D1,
    const i32 D2,
    const i32 D3,
    const i32 D4,
    const i32 D5,
    const f32 scale,
    const f32 perturb_scale,
    const u32 base_seed,
    const u32 layer_id
) {
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    const i32 total = D1 * D2 * D3 * D4 * D5;
    
    if (idx >= total) return;
    
    const i32 row = idx / D5;
    const i32 col = idx % D5;
    
    f32 base_val = static_cast<f32>(base_weights[idx]) * scale;
    
    f32 perturb_sum = 0.0f;
    for (i32 b = 0; b < batch_size; b++) {
        f32 p = get_perturbation(base_seed, b, layer_id, row, col, perturb_scale);
        perturb_sum += softmax_weights[b] * p;
    }
    
    f32 new_val = base_val + perturb_sum;
    out_weights[idx] = quantize(new_val, 1.0f / scale);
}

// ============ Patchifier with Positional Encoding ============
// Image: [batch, 64, 64, 3] HWC layout
// Patches: 8x8 grid of 8x8x3=192 pixel patches
// Output: [batch, 64, out_dim] tokens

__global__ void patchify_and_project(
    const f32* __restrict__ images,        // [batch, 64, 64, 3]
    const i8*  __restrict__ proj_weights,  // [192, out_dim]
          i8*  __restrict__ tokens,        // [batch, 64, out_dim]
    const i32              batch_size,
    const i32              out_dim,
    const f32              scale,
    const f32              perturb_scale,
    const u32              base_seed,
    const u32              layer_id,
    const f32              pe_scale
) {
    // Grid: (4, batch_size) - 4 groups of 16 patches each
    // Block: 32 threads (1 warp)
    const i32 patch_group = blockIdx.x;  // 0-3, each handles 16 patches
    const i32 batch_idx   = blockIdx.y;
    const i32 tid         = threadIdx.x;
    
    const f32 scale_sq = scale * scale;
    const f32 inv_scale = 1.0f / scale;
    
    const i32 patch_start = patch_group * 16;  // Starting patch index (0, 16, 32, 48)
    const i32 num_out_chunks = out_dim / 16;   // Number of 16-wide output chunks
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, i8, wmma::row_major> patch_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, i8, wmma::row_major> weight_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, i32> acc_frag;
    
    __shared__ i8  patch_smem[256];   // 16 patches x 16 input dims
    __shared__ i8  weight_smem[256];  // 16 input dims x 16 output dims
    __shared__ i32 tmp_i32[256];
    __shared__ f32 accumulated[256];  // 16 patches x 16 output dims
    
    // Process each output chunk
    for (i32 out_chunk = 0; out_chunk < num_out_chunks; out_chunk++) {
        // Zero accumulator
        for (i32 i = tid; i < 256; i += 32) {
            accumulated[i] = 0.0f;
        }
        __syncthreads();
        
        // Process input in chunks of 16 (192 = 12 chunks of 16)
        for (i32 in_chunk = 0; in_chunk < 12; in_chunk++) {
            // Load patch data: 16 patches x 16 input dimensions
            // Input linearization: patch pixel (local_y, local_x, channel) -> index in 192
            // HWC: index = local_y * 24 + local_x * 3 + channel
            for (i32 i = tid; i < 256; i += 32) {
                const i32 local_patch = i / 16;  // Which of 16 patches (0-15)
                const i32 local_dim = i % 16;    // Which of 16 input dims in this chunk
                
                const i32 patch_idx = patch_start + local_patch;  // Global patch index (0-63)
                const i32 py = patch_idx / 8;  // Patch grid Y (0-7)
                const i32 px = patch_idx % 8;  // Patch grid X (0-7)
                
                // Convert linear index in 192-dim patch to (local_y, local_x, channel)
                const i32 flat_idx = in_chunk * 16 + local_dim;  // 0-191
                const i32 local_y = flat_idx / 24;       // 0-7 (8 rows, each 24 values)
                const i32 local_x = (flat_idx % 24) / 3; // 0-7 (8 cols)
                const i32 channel = flat_idx % 3;        // 0-2 (RGB)
                
                // Global image coordinates
                const i32 img_y = py * 8 + local_y;
                const i32 img_x = px * 8 + local_x;
                
                const f32 pixel = images[batch_idx * 64 * 64 * 3 + img_y * 64 * 3 + img_x * 3 + channel];
                patch_smem[i] = quantize(pixel, inv_scale);
            }
            
            // Load weights with perturbation
            for (i32 i = tid; i < 256; i += 32) {
                const i32 row = i / 16;  // Input dim within chunk (0-15)
                const i32 col = i % 16;  // Output dim within chunk (0-15)
                const i32 weight_row = in_chunk * 16 + row;      // Global input dim (0-191)
                const i32 weight_col = out_chunk * 16 + col;     // Global output dim
                
                f32 base_w = static_cast<f32>(proj_weights[weight_row * out_dim + weight_col]) * scale;
                f32 perturb = get_perturbation(base_seed, batch_idx, layer_id, weight_row, weight_col, perturb_scale);
                weight_smem[i] = quantize(base_w + perturb, inv_scale);
            }
            __syncthreads();
            
            // Matrix multiply: [16 patches x 16 in] @ [16 in x 16 out] = [16 patches x 16 out]
            wmma::load_matrix_sync(patch_frag, patch_smem, 16);
            wmma::load_matrix_sync(weight_frag, weight_smem, 16);
            wmma::fill_fragment(acc_frag, 0);
            wmma::mma_sync(acc_frag, patch_frag, weight_frag, acc_frag);
            wmma::store_matrix_sync(tmp_i32, acc_frag, 16, wmma::mem_row_major);
            __syncthreads();
            
            // Accumulate
            for (i32 i = tid; i < 256; i += 32) {
                accumulated[i] += static_cast<f32>(tmp_i32[i]) * scale_sq;
            }
            __syncthreads();
        }
        
        // Add positional encoding and write output
        const i32 token_offset = batch_idx * 64 * out_dim + patch_start * out_dim + out_chunk * 16;
        for (i32 i = tid; i < 256; i += 32) {
            const i32 local_patch = i / 16;
            const i32 local_dim = i % 16;
            const i32 patch_idx = patch_start + local_patch;
            const i32 py = patch_idx / 8;
            const i32 px = patch_idx % 8;
            const i32 dim = out_chunk * 16 + local_dim;
            
            // Add 2D sinusoidal positional encoding
            f32 pe = sinusoidal_pe_2d(py, px, dim, out_dim) * pe_scale;
            f32 val_with_pe = accumulated[i] + pe;
            
            tokens[token_offset + local_patch * out_dim + local_dim] = quantize(val_with_pe, inv_scale);
        }
        __syncthreads();
    }
}

// ============ Final Projection ============

__global__ void final_projection(
    const i8* __restrict__ tokens,
    const i8* __restrict__ proj_weights,
          i8* __restrict__ output,
    const i32              batch_size,
    const f32              scale,
    const f32              perturb_scale,
    const u32              base_seed,
    const u32              layer_id
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
        const i32 row_base = out_chunk * 16;
        
        for (i32 i = tid; i < 256; i += 32) {
            const i32 row = i / 16;
            const i32 col = i % 16;
            
            f32 base_w = static_cast<f32>(proj_weights[out_chunk * 256 + i]) * scale;
            f32 perturb = get_perturbation(base_seed, batch_idx, layer_id, row_base + row, col, perturb_scale);
            weight_smem[i] = quantize(base_w + perturb, inv_scale);
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
            f32 val = static_cast<f32>(tmp_i32[i]) * scale_sq;
            f32 tanh_val = tanhf(val);
            output[out_offset + local_tok * 64 + local_dim] = static_cast<i8>(rintf(tanh_val * 127.0f));
        }
        __syncthreads();
    }
}

// ============ Materialize Image ============

__global__ void materialize_image(
    const i8* __restrict__ patch_weights,
    const i8* __restrict__ learnable_patches,
          i8* __restrict__ output,
    const i32              batch_size,
    const f32              perturb_scale,
    const u32              base_seed,
    const u32              layer_id
) {
    const i32 pos_idx   = blockIdx.x;
    const i32 batch_idx = blockIdx.y;
    const i32 tid       = threadIdx.x;
    
    const i32 py = pos_idx / 8;
    const i32 px = pos_idx % 8;
    
    __shared__ f32 weights[64];
    for (i32 i = tid; i < 64; i += 32) {
        weights[i] = static_cast<f32>(patch_weights[batch_idx * 64 * 64 + pos_idx * 64 + i]) / 127.0f;
    }
    __syncthreads();
    
    for (i32 pixel_idx = tid; pixel_idx < 192; pixel_idx += 32) {
        const i32 local_y = pixel_idx / 24;
        const i32 local_x = (pixel_idx % 24) / 3;
        const i32 channel = pixel_idx % 3;
        
        f32 blended = 0.0f;
        for (i32 p = 0; p < 64; p++) {
            f32 patch_pixel = static_cast<f32>(
                learnable_patches[p * 192 + local_y * 24 + local_x * 3 + channel]
            );
            patch_pixel += get_perturbation(base_seed, batch_idx, layer_id, p, pixel_idx, perturb_scale);
            blended += patch_pixel * weights[p];
        }
        
        const i32 img_y = py * 8 + local_y;
        const i32 img_x = px * 8 + local_x;
        output[batch_idx * 64 * 64 * 3 + img_y * 64 * 3 + img_x * 3 + channel] = 
            static_cast<i8>(min(max(rintf(blended), -127.0f), 127.0f));
    }
}

// ============ QKV Projection ============

__global__ void compute_qkv_batched(
    const i8* __restrict__ tokens,
    const i8* __restrict__ QKV_weights,
          i8* __restrict__ Q_output,
          i8* __restrict__ K_output,
          i8* __restrict__ V_output,
    const i32              batch_size,
    const i32              num_tokens,
    const i32              num_heads,
    const f32              scale,
    const f32              perturb_scale,
    const u32              base_seed,
    const u32              layer_id
) {
    const i32 token_block = blockIdx.x;
    const i32 head_idx    = blockIdx.y;
    const i32 batch_idx   = blockIdx.z;
    const i32 tid         = threadIdx.x;
    
    const f32 scale_sq  = scale * scale;
    const f32 inv_scale = 1.0f / scale;
    
    const i32 in_dim = num_heads * 16;
    const i32 tokens_offset = batch_idx * num_tokens * in_dim + token_block * 16 * in_dim;
    const i32 output_offset = batch_idx * num_heads * num_tokens * 16 
                            + head_idx * num_tokens * 16 
                            + token_block * 256;
    
    const i32 weight_base = head_idx * 3 * num_heads * 256;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, i8, wmma::row_major> tok_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, i8, wmma::row_major> weight_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, i32> acc_frag;
    
    __shared__ i8  tok_smem[256];
    __shared__ i8  weight_smem[256];
    __shared__ i32 tmp_i32[256];
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
        
        // Q weights
        const i32 q_weight_offset = weight_base + in_head * 256;
        const i32 q_row_base = ((head_idx * 3 + 0) * num_heads + in_head) * 16;
        for (i32 i = tid; i < 256; i += 32) {
            const i32 row = i / 16;
            const i32 col = i % 16;
            f32 base_w = static_cast<f32>(QKV_weights[q_weight_offset + i]) * scale;
            f32 perturb = get_perturbation(base_seed, batch_idx, layer_id, q_row_base + row, col, perturb_scale);
            weight_smem[i] = quantize(base_w + perturb, inv_scale);
        }
        __syncthreads();
        
        wmma::load_matrix_sync(weight_frag, weight_smem, 16);
        wmma::fill_fragment(acc_frag, 0);
        wmma::mma_sync(acc_frag, tok_frag, weight_frag, acc_frag);
        wmma::store_matrix_sync(tmp_i32, acc_frag, 16, wmma::mem_row_major);
        __syncthreads();
        
        for (i32 i = tid; i < 256; i += 32) {
            acc_Q[i] += static_cast<f32>(tmp_i32[i]) * scale_sq;
        }
        __syncthreads();
        
        // K weights
        const i32 k_weight_offset = weight_base + num_heads * 256 + in_head * 256;
        const i32 k_row_base = ((head_idx * 3 + 1) * num_heads + in_head) * 16;
        for (i32 i = tid; i < 256; i += 32) {
            const i32 row = i / 16;
            const i32 col = i % 16;
            f32 base_w = static_cast<f32>(QKV_weights[k_weight_offset + i]) * scale;
            f32 perturb = get_perturbation(base_seed, batch_idx, layer_id, k_row_base + row, col, perturb_scale);
            weight_smem[i] = quantize(base_w + perturb, inv_scale);
        }
        __syncthreads();
        
        wmma::load_matrix_sync(weight_frag, weight_smem, 16);
        wmma::fill_fragment(acc_frag, 0);
        wmma::mma_sync(acc_frag, tok_frag, weight_frag, acc_frag);
        wmma::store_matrix_sync(tmp_i32, acc_frag, 16, wmma::mem_row_major);
        __syncthreads();
        
        for (i32 i = tid; i < 256; i += 32) {
            acc_K[i] += static_cast<f32>(tmp_i32[i]) * scale_sq;
        }
        __syncthreads();
        
        // V weights
        const i32 v_weight_offset = weight_base + 2 * num_heads * 256 + in_head * 256;
        const i32 v_row_base = ((head_idx * 3 + 2) * num_heads + in_head) * 16;
        for (i32 i = tid; i < 256; i += 32) {
            const i32 row = i / 16;
            const i32 col = i % 16;
            f32 base_w = static_cast<f32>(QKV_weights[v_weight_offset + i]) * scale;
            f32 perturb = get_perturbation(base_seed, batch_idx, layer_id, v_row_base + row, col, perturb_scale);
            weight_smem[i] = quantize(base_w + perturb, inv_scale);
        }
        __syncthreads();
        
        wmma::load_matrix_sync(weight_frag, weight_smem, 16);
        wmma::fill_fragment(acc_frag, 0);
        wmma::mma_sync(acc_frag, tok_frag, weight_frag, acc_frag);
        wmma::store_matrix_sync(tmp_i32, acc_frag, 16, wmma::mem_row_major);
        __syncthreads();
        
        for (i32 i = tid; i < 256; i += 32) {
            acc_V[i] += static_cast<f32>(tmp_i32[i]) * scale_sq;
        }
        __syncthreads();
    }
    
    for (i32 i = tid; i < 256; i += 32) {
        Q_output[output_offset + i] = quantize(acc_Q[i], inv_scale);
        K_output[output_offset + i] = quantize(acc_K[i], inv_scale);
        V_output[output_offset + i] = quantize(acc_V[i], inv_scale);
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
    const f32              scale,
    const f32              perturb_scale,
    const u32              base_seed,
    const u32              layer_id
) {
    const i32 token_block = blockIdx.x;
    const i32 batch_idx   = blockIdx.y;
    const i32 tid         = threadIdx.x;
    
    const f32 scale_sq  = scale * scale;
    const f32 inv_scale = 1.0f / scale;
    
    const i32 out_dim = num_heads * 16;
    const i32 residual_offset = batch_idx * num_tokens * out_dim + token_block * 16 * out_dim;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, i8, wmma::row_major> attn_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, i8, wmma::row_major> proj_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, i32> acc_frag;
    
    __shared__ i8  attn_smem[256];
    __shared__ i8  proj_smem[256];
    __shared__ i32 tmp_i32[256];
    __shared__ f32 accumulated[256];
    
    for (i32 head_idx = 0; head_idx < num_heads; head_idx++) {
        for (i32 i = tid; i < 256; i += 32) {
            const i32 local_tok = i / 16;
            const i32 local_dim = i % 16;
            accumulated[i] = static_cast<f32>(
                residual[residual_offset + local_tok * out_dim + head_idx * 16 + local_dim]
            ) * scale;
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
            f32 base_w = static_cast<f32>(proj_weights[proj_offset + i]) * scale;
            f32 perturb = get_perturbation(base_seed, batch_idx, layer_id, proj_row_base + row, col, perturb_scale);
            proj_smem[i] = quantize(base_w + perturb, inv_scale);
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
        
        for (i32 i = tid; i < 256; i += 32) {
            const i32 local_tok = i / 16;
            const i32 local_dim = i % 16;
            output[residual_offset + local_tok * out_dim + head_idx * 16 + local_dim] = 
                quantize(accumulated[i], inv_scale);
        }
        __syncthreads();
    }
}

// ============ Host functions ============

torch::Tensor patchify_fn(
    int64_t stream_ptr,
    torch::Tensor images,
    torch::Tensor proj_weights,
    double max_val,
    double perturb_scale,
    int64_t seed,
    int64_t layer_id,
    double pe_scale
) {
    const i32 batch_size = images.size(0);
    const i32 out_dim = proj_weights.size(1);
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
        out_dim,
        scale,
        static_cast<f32>(perturb_scale),
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
    double max_val,
    double perturb_scale,
    int64_t seed,
    int64_t layer_id
) {
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
        scale,
        static_cast<f32>(perturb_scale),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
    );
    
    return output;
}

torch::Tensor materialize_image_fn(
    int64_t stream_ptr,
    torch::Tensor patch_weights,
    torch::Tensor learnable_patches,
    double perturb_scale,
    int64_t seed,
    int64_t layer_id
) {
    const i32 batch_size = patch_weights.size(0);
    
    auto output = torch::empty({batch_size, 64, 64, 3}, 
                               torch::dtype(torch::kInt8).device(patch_weights.device()));
    
    dim3 grid(64, batch_size);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    materialize_image<<<grid, 32, 0, stream>>>(
        patch_weights.data_ptr<i8>(),
        learnable_patches.data_ptr<i8>(),
        output.data_ptr<i8>(),
        batch_size,
        static_cast<f32>(perturb_scale),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
    );
    
    return output;
}

std::vector<torch::Tensor> compute_qkv_batched_fn(
    int64_t stream_ptr,
    torch::Tensor tokens,
    torch::Tensor QKV_weights,
    double max_val,
    double perturb_scale,
    int64_t seed,
    int64_t layer_id
) {
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
        scale,
        static_cast<f32>(perturb_scale),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
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
    double max_val,
    double perturb_scale,
    int64_t seed,
    int64_t layer_id
) {
    const i32 batch_size = attn_output.size(0);
    const i32 num_heads  = attn_output.size(1);
    const i32 num_tokens = attn_output.size(2);
    const f32 scale      = static_cast<f32>(max_val) / 127.0f;
    
    auto output = torch::empty({batch_size, num_tokens, num_heads * 16}, 
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
        scale,
        static_cast<f32>(perturb_scale),
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
        output.data_ptr<i8>(),
        ground_truth.data_ptr<i8>(),
        scores.data_ptr<f32>(),
        batch_size
    );
    
    return scores;
}

torch::Tensor compute_softmax_fn(
    int64_t stream_ptr,
    torch::Tensor scores,
    double temperature
) {
    const i32 batch_size = scores.size(0);
    
    auto softmax_out = torch::empty({batch_size}, 
                                    torch::dtype(torch::kFloat32).device(scores.device()));
    
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    compute_softmax<<<1, 256, 0, stream>>>(
        scores.data_ptr<f32>(),
        softmax_out.data_ptr<f32>(),
        batch_size,
        static_cast<f32>(temperature)
    );
    
    return softmax_out;
}

torch::Tensor accumulate_perturbation_2d_fn(
    int64_t stream_ptr,
    torch::Tensor base_weights,
    torch::Tensor softmax_weights,
    double max_val,
    double perturb_scale,
    int64_t seed,
    int64_t layer_id
) {
    const i32 batch_size = softmax_weights.size(0);
    const i32 M = base_weights.size(0);
    const i32 N = base_weights.size(1);
    const i32 total = M * N;
    const f32 scale = static_cast<f32>(max_val) / 127.0f;
    
    auto out_weights = torch::empty_like(base_weights);
    
    const i32 threads = 256;
    const i32 blocks = (total + threads - 1) / threads;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    accumulate_perturbation_2d<<<blocks, threads, 0, stream>>>(
        base_weights.data_ptr<i8>(),
        out_weights.data_ptr<i8>(),
        softmax_weights.data_ptr<f32>(),
        batch_size,
        M,
        N,
        scale,
        static_cast<f32>(perturb_scale),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
    );
    
    return out_weights;
}

torch::Tensor accumulate_perturbation_3d_fn(
    int64_t stream_ptr,
    torch::Tensor base_weights,
    torch::Tensor softmax_weights,
    double max_val,
    double perturb_scale,
    int64_t seed,
    int64_t layer_id
) {
    const i32 batch_size = softmax_weights.size(0);
    const i32 D1 = base_weights.size(0);
    const i32 D2 = base_weights.size(1);
    const i32 D3 = base_weights.size(2);
    const i32 total = D1 * D2 * D3;
    const f32 scale = static_cast<f32>(max_val) / 127.0f;
    
    auto out_weights = torch::empty_like(base_weights);
    
    const i32 threads = 256;
    const i32 blocks = (total + threads - 1) / threads;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    accumulate_perturbation_3d<<<blocks, threads, 0, stream>>>(
        base_weights.data_ptr<i8>(),
        out_weights.data_ptr<i8>(),
        softmax_weights.data_ptr<f32>(),
        batch_size,
        D1,
        D2,
        D3,
        scale,
        static_cast<f32>(perturb_scale),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
    );
    
    return out_weights;
}

torch::Tensor accumulate_perturbation_5d_fn(
    int64_t stream_ptr,
    torch::Tensor base_weights,
    torch::Tensor softmax_weights,
    double max_val,
    double perturb_scale,
    int64_t seed,
    int64_t layer_id
) {
    const i32 batch_size = softmax_weights.size(0);
    const i32 D1 = base_weights.size(0);
    const i32 D2 = base_weights.size(1);
    const i32 D3 = base_weights.size(2);
    const i32 D4 = base_weights.size(3);
    const i32 D5 = base_weights.size(4);
    const i32 total = D1 * D2 * D3 * D4 * D5;
    const f32 scale = static_cast<f32>(max_val) / 127.0f;
    
    auto out_weights = torch::empty_like(base_weights);
    
    const i32 threads = 256;
    const i32 blocks = (total + threads - 1) / threads;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    accumulate_perturbation_5d<<<blocks, threads, 0, stream>>>(
        base_weights.data_ptr<i8>(),
        out_weights.data_ptr<i8>(),
        softmax_weights.data_ptr<f32>(),
        batch_size,
        D1,
        D2,
        D3,
        D4,
        D5,
        scale,
        static_cast<f32>(perturb_scale),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
    );
    
    return out_weights;
}
;//
"""

cpp_source = """
//js
torch::Tensor patchify_fn(int64_t stream_ptr, torch::Tensor images, torch::Tensor proj_weights, double max_val, double perturb_scale, int64_t seed, int64_t layer_id, double pe_scale);
torch::Tensor final_projection_fn(int64_t stream_ptr, torch::Tensor tokens, torch::Tensor proj_weights, double max_val, double perturb_scale, int64_t seed, int64_t layer_id);
torch::Tensor materialize_image_fn(int64_t stream_ptr, torch::Tensor patch_weights, torch::Tensor learnable_patches, double perturb_scale, int64_t seed, int64_t layer_id);
std::vector<torch::Tensor> compute_qkv_batched_fn(int64_t stream_ptr, torch::Tensor tokens, torch::Tensor QKV_weights, double max_val, double perturb_scale, int64_t seed, int64_t layer_id);
torch::Tensor flash_attention_batched_fn(int64_t stream_ptr, torch::Tensor Q, torch::Tensor K, torch::Tensor V, double max_val);
torch::Tensor project_and_residual_fn(int64_t stream_ptr, torch::Tensor attn_output, torch::Tensor proj_weights, torch::Tensor residual, double max_val, double perturb_scale, int64_t seed, int64_t layer_id);
torch::Tensor compute_mse_scores_fn(int64_t stream_ptr, torch::Tensor output, torch::Tensor ground_truth);
torch::Tensor compute_softmax_fn(int64_t stream_ptr, torch::Tensor scores, double temperature);
torch::Tensor accumulate_perturbation_2d_fn(int64_t stream_ptr, torch::Tensor base_weights, torch::Tensor softmax_weights, double max_val, double perturb_scale, int64_t seed, int64_t layer_id);
torch::Tensor accumulate_perturbation_3d_fn(int64_t stream_ptr, torch::Tensor base_weights, torch::Tensor softmax_weights, double max_val, double perturb_scale, int64_t seed, int64_t layer_id);
torch::Tensor accumulate_perturbation_5d_fn(int64_t stream_ptr, torch::Tensor base_weights, torch::Tensor softmax_weights, double max_val, double perturb_scale, int64_t seed, int64_t layer_id);
;//
"""

module = load_inline(
    name="vit_int8_evolutionary_v10",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=[
        "patchify_fn", "final_projection_fn", "materialize_image_fn",
        "compute_qkv_batched_fn", "flash_attention_batched_fn", "project_and_residual_fn",
        "compute_mse_scores_fn", "compute_softmax_fn",
        "accumulate_perturbation_2d_fn", "accumulate_perturbation_3d_fn", "accumulate_perturbation_5d_fn"
    ],
    verbose=True,
    extra_cuda_cflags=["-O3", "-arch=sm_80"]
)

def quantize(x, max_val):
    scale = max_val / 127.0
    return (x / scale).round().clamp(-127, 127).to(torch.int8)

def dequantize(q, max_val):
    return q.float() * (max_val / 127.0)

# ============ Model Weights Container ============
class ViTWeights:
    def __init__(self, num_heads, num_layers, max_val, device="cuda"):
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_val = max_val
        self.device = device
        
        out_dim = num_heads * 16
        patchifier_f = torch.randn(192, out_dim, device=device) / math.sqrt(192)
        self.patchifier = quantize(patchifier_f, max_val)
        
        final_proj_f = torch.randn(4, 16, 16, device=device) / math.sqrt(16)
        self.final_proj = quantize(final_proj_f, max_val)
        
        patches_f = torch.randn(64, 8, 8, 3, device=device) * 40
        self.learnable_patches = patches_f.round().clamp(-127, 127).to(torch.int8)
        
        self.qkv_weights = []
        self.proj_weights = []
        for _ in range(num_layers):
            qkv_f = torch.randn(num_heads, 3, num_heads, 16, 16, device=device) / math.sqrt(16 * num_heads)
            self.qkv_weights.append(quantize(qkv_f, max_val))
            
            proj_f = torch.randn(num_heads, 16, 16, device=device) / math.sqrt(16 * num_heads)
            self.proj_weights.append(quantize(proj_f, max_val))
    
    def accumulate_perturbations(self, softmax_weights, seed, perturb_scale, stream):
        """Apply weighted perturbations to all weights."""
        self.patchifier = module.accumulate_perturbation_2d_fn(
            stream, self.patchifier, softmax_weights, 
            self.max_val, perturb_scale, seed, 0
        )
        
        self.final_proj = module.accumulate_perturbation_3d_fn(
            stream, self.final_proj, softmax_weights,
            self.max_val, perturb_scale, seed, 1
        )
        
        patches_flat = self.learnable_patches.reshape(64, -1)
        patches_updated = module.accumulate_perturbation_2d_fn(
            stream, patches_flat, softmax_weights,
            self.max_val, perturb_scale, seed, 2
        )
        self.learnable_patches = patches_updated.reshape(64, 8, 8, 3)
        
        for i in range(self.num_layers):
            layer_base = 100 + i * 10
            
            self.qkv_weights[i] = module.accumulate_perturbation_5d_fn(
                stream, self.qkv_weights[i], softmax_weights,
                self.max_val, perturb_scale, seed, layer_base
            )
            
            self.proj_weights[i] = module.accumulate_perturbation_3d_fn(
                stream, self.proj_weights[i], softmax_weights,
                self.max_val, perturb_scale, seed, layer_base + 1
            )

# ============ Forward Pass ============
def forward_pass(images, weights, seed, perturb_scale, pe_scale, stream):
    """Run full forward pass with perturbations and positional encoding."""
    max_val = weights.max_val
    
    # 1. Patchify with positional encoding
    tokens = module.patchify_fn(
        stream, images, weights.patchifier, 
        max_val, perturb_scale, seed, 0, pe_scale
    )
    
    # 2. Attention layers
    x = tokens
    for i, (qkv_w, proj_w) in enumerate(zip(weights.qkv_weights, weights.proj_weights)):
        layer_base = 100 + i * 10
        
        Q, K, V = module.compute_qkv_batched_fn(
            stream, x, qkv_w, max_val, perturb_scale, seed, layer_base
        )
        
        attn_out = module.flash_attention_batched_fn(stream, Q, K, V, max_val)
        
        x = module.project_and_residual_fn(
            stream, attn_out, proj_w, x, max_val, perturb_scale, seed, layer_base + 1
        )
    
    # 3. Reduce to [batch, 64, 16]
    x_reduced = x[:, :, :16].contiguous()
    
    # 4. Final projection
    patch_weights = module.final_projection_fn(
        stream, x_reduced, weights.final_proj,
        max_val, perturb_scale, seed, 1
    )
    
    # 5. Materialize image
    output = module.materialize_image_fn(
        stream, patch_weights, weights.learnable_patches,
        perturb_scale, seed, 2
    )
    
    return output

# ============ Training Loop ============
def train_step(images, ground_truth, weights, seed, perturb_scale, pe_scale, temperature, stream):
    """Single training step."""
    output = forward_pass(images, weights, seed, perturb_scale, pe_scale, stream)
    scores = module.compute_mse_scores_fn(stream, output, ground_truth)
    softmax_weights = module.compute_softmax_fn(stream, scores, temperature)
    weights.accumulate_perturbations(softmax_weights, seed, perturb_scale, stream)
    return output, scores, softmax_weights

# ============ Test ============
print("=" * 60)
print("Testing Evolutionary ViT Training with Positional Encoding")
print("=" * 60)

batch_size      = 256
num_heads       = 8
num_layers      = 8
max_val         = 6.0
perturb_scale   = 0.15
pe_scale        = 1.0
temperature     = 0.00001

# torch.manual_seed(123)
stream = torch.cuda.Stream()

weights = ViTWeights(num_heads, num_layers, max_val)

# Low frequency target

class CatDataloader:
    def __init__(self, path):
        self.paths = []
        self.data  = []
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        for file in os.listdir(path):
            if file.endswith(".png"):
                self.paths.append(os.path.join(path, file))

        
    def next(self, batch_size):
        batch = set()

        def aug(image):
            if random.random() < 0.5:
                # Apply random augmentations
                if random.random() < 0.5:
                    image = torch.flip(image, [3]) # horizontal flip
                # if random.random() < 0.5:
                #     image = torch.flip(image, [2]) # vertical flip
                # angle = random.choice([0, 90, 180, 270])
                # if angle != 0:
                #     image = torch.rot90(image, k=angle // 90, dims=[2, 3])
                
                # if random.random() < 0.5:
                #     # zoom
                #     scale = random.uniform(1.0, 2.2)
                #     new_size = int(64 * scale)
                #     image = F.interpolate(image, size=(new_size, new_size), mode='bicubic', align_corners=False)
                #     crop_x = random.randint(0, new_size - 64)
                #     crop_y = random.randint(0, new_size - 64)
                #     image = image[:, :, crop_x:crop_x+64, crop_y:crop_y+64]

            return image
        while len(batch) < batch_size:
            # Load a new image with a chance, otherwise reuse the old one
            prob = 0.5
            if len(self.paths) > 0 and (random.random() < prob or len(self.data) < 64):
                img_path = random.choice(range(len(self.paths)))
                image    = Image.open(self.paths[img_path])
                image    = self.transform(image).unsqueeze(0) # Map to [B, C, H, W]
                self.paths.remove(self.paths[img_path])
                self.data.append(image)
                batch.add(aug(image))

            else:

                image = random.choice(self.data)
                image = aug(image)
                batch.add(image)

        batch = list(set(batch))

        return torch.cat(batch, dim=0)

dataset = CatDataloader("data\\MNIST\\dataset-part1")

assert dataset.next(1).shape == (1, 3, 64, 64), f"Unexpected shape: {dataset.next(1).shape}"
size=64

def assemble_batch():
    batch = dataset.next(batch_size=batch_size)
    return batch.to("cuda").permute(0, 2, 3, 1)  # Map to [B, H, W, C]


# ground_truth_f = torch.randn(1, 4, 4, 3, device="cuda")
ground_truth_f = assemble_batch()[0:1, :, :, :] # Use one image
# ground_truth_f = torch.nn.functional.interpolate(
# ground_truth_f.permute(0, 3, 1, 2), size=(64, 64), mode='bilinear', align_corners=False
# ).permute(0, 2, 3, 1)
ground_truth_single = (ground_truth_f * 127.0).round().clamp(-127, 127).to(torch.int8)
ground_truth = ground_truth_single.expand(batch_size, -1, -1, -1).contiguous()

print(f"\nConfig:")
print(f"  Batch size: {batch_size}")
print(f"  Num heads: {num_heads}")
print(f"  Num layers: {num_layers}")
print(f"  Perturb scale: {perturb_scale}")
print(f"  PE scale: {pe_scale}")
print(f"  Temperature: {temperature}")

num_epochs = 1 << 12
losses = []

for epoch in range(num_epochs):
    seed = epoch * 1000
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record(stream)
    
    # sigmal_noise = torch.rand(batch_size, 1, 1, 1, device="cuda")
    sigmal_noise = 0.05
    images     = ground_truth * (1.0 - sigmal_noise) + sigmal_noise * torch.rand(batch_size, 64, 64, 3, device="cuda")

    output, scores, softmax_weights = train_step(
        images, ground_truth, weights, seed, perturb_scale, pe_scale, temperature, stream.cuda_stream
    )
    
    end_event.record(stream)
    end_event.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    
    scores_np = scores.cpu().numpy()
    softmax_np = softmax_weights.cpu().numpy()
    
    best_mse = -scores_np.max()
    mean_mse = -scores_np.mean()
    losses.append(best_mse)


    entropy = -(softmax_np * np.log(softmax_np + 1e-10)).sum()
    
    if epoch % 20 == 0 or epoch == num_epochs - 1:
        try:
            dds = dds_from_tensor(output[0:1, :, :, :].permute(0, 3, 1, 2))
            dds.save(".tmp/epoch.dds")
        except:
            pass
        print(f"\nEpoch {epoch + 1}:")
        print(f"  Time: {elapsed_ms:.2f} ms")
        print(f"  Best MSE: {best_mse:.2f}, Mean MSE: {mean_mse:.2f}")
        print(f"  Softmax entropy: {entropy:.2f} (max={np.log(batch_size):.2f})")
        print(f"  Top 5 weights: {np.sort(softmax_np)[-5:][::-1].round(3)}")

print(f"\n" + "=" * 60)
print(f"Loss progression: {losses[0]:.2f} -> {losses[-1]:.2f}")
print(f"Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
print("=" * 60)
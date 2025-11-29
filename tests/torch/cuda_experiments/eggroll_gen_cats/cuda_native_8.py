# MIT License
# Copyright (c) 2025 Anton Schreiner

import math
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline
import os
from PIL import Image
import torchvision.transforms as transforms
import random
from py.torch_utils import dds_from_tensor

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
    u32 base_seed, i32 batch_idx, u32 layer_id, i32 row, i32 col, f32 scale, i32 norm_size
) {
    const f32 norm = sqrtf((f32)norm_size);
    u32 u_seed = base_seed ^ (batch_idx * 12345) ^ (layer_id * 111111) ^ (row * 67890);
    u32 v_seed = base_seed ^ (batch_idx * 12345) ^ (layer_id * 111111) ^ ((col + 100000) * 67890);
    
    f32 u_val = random_normal(u_seed) * scale;
    f32 v_val = random_normal(v_seed) * scale;
    
    return u_val * v_val / norm;
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

// ============ Scoring Kernels ============

__global__ void compute_mse_scores(
    const f32* __restrict__ output,
    const f32* __restrict__ ground_truth,
          f32* __restrict__ scores,
    const i32 batch_size
) {
    const i32 batch_idx = blockIdx.x;
    const i32 tid = threadIdx.x;
    
    __shared__ f32 partial_sum[256];
    
    f32 local_sum = 0.0f;
    const i32 num_pixels = 64 * 64 * 3;
    
    for (i32 i = tid; i < num_pixels; i += blockDim.x) {
        f32 diff = output[batch_idx * num_pixels + i] - ground_truth[i];
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

// ============ Accumulate Perturbations ============

__global__ void accumulate_perturbation_2d(
          f32* __restrict__ weights,
    const f32* __restrict__ softmax_weights,
    const i32 batch_size,
    const i32 M,
    const i32 N,
    const f32 perturb_scale,
    const u32 base_seed,
    const u32 layer_id,
    const i32 norm_size
) {
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    const i32 total = M * N;
    
    if (idx >= total) return;
    
    const i32 row = idx / N;
    const i32 col = idx % N;
    
    f32 perturb_sum = 0.0f;
    for (i32 b = 0; b < batch_size; b++) {
        f32 p = get_perturbation(base_seed, b, layer_id, row, col, perturb_scale, norm_size);
        perturb_sum += sqrtf(softmax_weights[b]) * p;
    }
    
    weights[idx] += perturb_sum;
}

// ============ Patchifier ============

__global__ void patchify_and_project(
    const f32* __restrict__ images,
    const f32* __restrict__ proj_weights,
          f32* __restrict__ tokens,
    const i32              batch_size,
    const i32              out_dim,
    const f32              perturb_scale,
    const u32              base_seed,
    const u32              layer_id,
    const f32              pe_scale
) {
    const i32 patch_idx = blockIdx.x;
    const i32 batch_idx = blockIdx.y;
    const i32 tid       = threadIdx.x;
    
    const i32 py = patch_idx / 8;
    const i32 px = patch_idx % 8;
    
    // Each thread handles some output dimensions
    for (i32 out_d = tid; out_d < out_dim; out_d += blockDim.x) {
        f32 sum = 0.0f;
        
        // Dot product over 192 input dimensions
        for (i32 in_d = 0; in_d < 192; in_d++) {
            const i32 local_y = in_d / 24;
            const i32 local_x = (in_d % 24) / 3;
            const i32 channel = in_d % 3;
            
            const i32 img_y = py * 8 + local_y;
            const i32 img_x = px * 8 + local_x;
            
            f32 pixel = images[batch_idx * 64 * 64 * 3 + img_y * 64 * 3 + img_x * 3 + channel];
            f32 weight = proj_weights[in_d * out_dim + out_d];
            weight += get_perturbation(base_seed, batch_idx, layer_id, in_d, out_d, perturb_scale, 192);
            
            sum += pixel * weight;
        }
        
        // Add positional encoding
        f32 pe = sinusoidal_pe_2d(py, px, out_d, out_dim) * pe_scale;
        tokens[batch_idx * 64 * out_dim + patch_idx * out_dim + out_d] = sum + pe;
    }
}

// ============ QKV Projection ============

__global__ void compute_qkv(
    const f32* __restrict__ tokens,
    const f32* __restrict__ QKV_weights,
          f32* __restrict__ Q_output,
          f32* __restrict__ K_output,
          f32* __restrict__ V_output,
    const i32              batch_size,
    const i32              num_tokens,
    const i32              in_dim,
    const i32              head_dim,
    const i32              num_heads,
    const f32              perturb_scale,
    const u32              base_seed,
    const u32              layer_id
) {
    const i32 token_idx = blockIdx.x;
    const i32 head_idx  = blockIdx.y;
    const i32 batch_idx = blockIdx.z;
    const i32 tid       = threadIdx.x;
    
    // Each thread computes some dimensions of Q, K, V for this token/head
    for (i32 d = tid; d < head_dim; d += blockDim.x) {
        f32 q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;
        
        for (i32 in_d = 0; in_d < in_dim; in_d++) {
            f32 tok = tokens[batch_idx * num_tokens * in_dim + token_idx * in_dim + in_d];
            
            // Weight layout: [num_heads, 3, in_dim, head_dim]
            const i32 q_idx = ((head_idx * 3 + 0) * in_dim + in_d) * head_dim + d;
            const i32 k_idx = ((head_idx * 3 + 1) * in_dim + in_d) * head_dim + d;
            const i32 v_idx = ((head_idx * 3 + 2) * in_dim + in_d) * head_dim + d;
            
            f32 q_w = QKV_weights[q_idx] + get_perturbation(base_seed, batch_idx, layer_id, 
                (head_idx * 3 + 0) * in_dim + in_d, d, perturb_scale, in_dim);
            f32 k_w = QKV_weights[k_idx] + get_perturbation(base_seed, batch_idx, layer_id,
                (head_idx * 3 + 1) * in_dim + in_d, d, perturb_scale, in_dim);
            f32 v_w = QKV_weights[v_idx] + get_perturbation(base_seed, batch_idx, layer_id,
                (head_idx * 3 + 2) * in_dim + in_d, d, perturb_scale, in_dim);
            
            q_sum += tok * q_w;
            k_sum += tok * k_w;
            v_sum += tok * v_w;
        }
        
        const i32 out_idx = batch_idx * num_heads * num_tokens * head_dim 
                          + head_idx * num_tokens * head_dim 
                          + token_idx * head_dim + d;
        Q_output[out_idx] = q_sum;
        K_output[out_idx] = k_sum;
        V_output[out_idx] = v_sum;
    }
}

// ============ Attention ============

__global__ void attention(
    const f32* __restrict__ Q,
    const f32* __restrict__ K,
    const f32* __restrict__ V,
          f32* __restrict__ O,
    const i32              batch_size,
    const i32              num_heads,
    const i32              num_tokens,
    const i32              head_dim
) {
    const i32 q_idx     = blockIdx.x;
    const i32 head_idx  = blockIdx.y;
    const i32 batch_idx = blockIdx.z;
    const i32 tid       = threadIdx.x;
    
    const f32 scale = 1.0f / sqrtf((f32)head_dim);
    
    __shared__ f32 scores[64];  // Assuming num_tokens <= 64
    __shared__ f32 max_score;
    __shared__ f32 sum_exp;
    
    // Compute attention scores for this query
    const i32 qo_base = batch_idx * num_heads * num_tokens * head_dim 
                      + head_idx * num_tokens * head_dim;
    
    for (i32 k_idx = tid; k_idx < num_tokens; k_idx += blockDim.x) {
        f32 dot = 0.0f;
        for (i32 d = 0; d < head_dim; d++) {
            f32 q_val = Q[qo_base + q_idx * head_dim + d];
            f32 k_val = K[qo_base + k_idx * head_dim + d];
            dot += q_val * k_val;
        }
        scores[k_idx] = dot * scale;
    }
    __syncthreads();
    
    // Softmax
    if (tid == 0) {
        max_score = scores[0];
        for (i32 i = 1; i < num_tokens; i++) {
            max_score = fmaxf(max_score, scores[i]);
        }
    }
    __syncthreads();
    
    for (i32 i = tid; i < num_tokens; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_score);
    }
    __syncthreads();
    
    if (tid == 0) {
        sum_exp = 0.0f;
        for (i32 i = 0; i < num_tokens; i++) {
            sum_exp += scores[i];
        }
    }
    __syncthreads();
    
    for (i32 i = tid; i < num_tokens; i += blockDim.x) {
        scores[i] /= sum_exp;
    }
    __syncthreads();
    
    // Weighted sum of values
    for (i32 d = tid; d < head_dim; d += blockDim.x) {
        f32 out_val = 0.0f;
        for (i32 v_idx = 0; v_idx < num_tokens; v_idx++) {
            out_val += scores[v_idx] * V[qo_base + v_idx * head_dim + d];
        }
        O[qo_base + q_idx * head_dim + d] = out_val;
    }
}

// ============ Output Projection with Residual ============

__global__ void project_and_residual(
    const f32* __restrict__ attn_output,
    const f32* __restrict__ proj_weights,
    const f32* __restrict__ residual,
          f32* __restrict__ output,
    const i32              batch_size,
    const i32              num_tokens,
    const i32              num_heads,
    const i32              head_dim,
    const f32              perturb_scale,
    const u32              base_seed,
    const u32              layer_id
) {
    const i32 token_idx = blockIdx.x;
    const i32 batch_idx = blockIdx.y;
    const i32 tid       = threadIdx.x;
    
    const i32 out_dim = num_heads * head_dim;
    
    // Each thread handles some output dimensions
    for (i32 out_d = tid; out_d < out_dim; out_d += blockDim.x) {
        f32 sum = 0.0f;
        
        // Project from all heads
        for (i32 h = 0; h < num_heads; h++) {
            for (i32 d = 0; d < head_dim; d++) {
                f32 attn_val = attn_output[batch_idx * num_heads * num_tokens * head_dim 
                                         + h * num_tokens * head_dim 
                                         + token_idx * head_dim + d];
                
                const i32 w_idx = (h * head_dim + d) * out_dim + out_d;
                f32 weight = proj_weights[w_idx];
                weight += get_perturbation(base_seed, batch_idx, layer_id, h * head_dim + d, out_d, perturb_scale, out_dim);
                
                sum += max(0.0f, attn_val) * weight;
            }
        }

        sum = sum + residual[batch_idx * num_tokens * out_dim + token_idx * out_dim + out_d];
        
        output[batch_idx * num_tokens * out_dim + token_idx * out_dim + out_d] = sum;
    }
}

// ============ Final Projection to Blend Weights ============

__global__ void final_projection(
    const f32* __restrict__ tokens,
    const f32* __restrict__ proj_weights,
          f32* __restrict__ output,
    const i32              batch_size,
    const i32              in_dim,
    const i32              out_dim,
    const f32              perturb_scale,
    const u32              base_seed,
    const u32              layer_id
) {
    const i32 token_idx = blockIdx.x;
    const i32 batch_idx = blockIdx.y;
    const i32 tid       = threadIdx.x;
    
    for (i32 out_d = tid; out_d < out_dim; out_d += blockDim.x) {
        f32 sum = 0.0f;
        
        for (i32 in_d = 0; in_d < in_dim; in_d++) {
            f32 tok = tokens[batch_idx * 64 * in_dim + token_idx * in_dim + in_d];
            f32 weight = proj_weights[in_d * out_dim + out_d];
            weight += get_perturbation(base_seed, batch_idx, layer_id, in_d, out_d, perturb_scale, in_dim);
            sum += tok * weight;
        }
        
        // Apply tanh to get blend weights in [-1, 1]
        output[batch_idx * 64 * out_dim + token_idx * out_dim + out_d] = tanhf(sum);
    }
}

// ============ Materialize Image ============

__global__ void materialize_image(
    const f32* __restrict__ blend_weights,
    const f32* __restrict__ learnable_patches,
          f32* __restrict__ output,
    const i32              batch_size,
    const i32              num_patches,
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
    for (i32 i = tid; i < num_patches; i += blockDim.x) {
        weights[i] = blend_weights[batch_idx * 64 * num_patches + pos_idx * num_patches + i];
    }
    __syncthreads();
    
    for (i32 pixel_idx = tid; pixel_idx < 192; pixel_idx += blockDim.x) {
        const i32 local_y = pixel_idx / 24;
        const i32 local_x = (pixel_idx % 24) / 3;
        const i32 channel = pixel_idx % 3;
        
        f32 blended = 0.0f;
        for (i32 p = 0; p < num_patches; p++) {
            f32 patch_pixel = learnable_patches[p * 192 + pixel_idx];
            patch_pixel += get_perturbation(base_seed, batch_idx, layer_id, p, pixel_idx, perturb_scale, 8);
            blended += patch_pixel * weights[p];
        }
        
        const i32 img_y = py * 8 + local_y;
        const i32 img_x = px * 8 + local_x;
        output[batch_idx * 64 * 64 * 3 + img_y * 64 * 3 + img_x * 3 + channel] = blended;
    }
}

// ============ Host Functions ============

torch::Tensor patchify_fn(
    torch::Tensor images,
    torch::Tensor proj_weights,
    int64_t batch_size,
    double perturb_scale,
    int64_t seed,
    int64_t layer_id,
    double pe_scale
) {
    const i32 out_dim = proj_weights.size(1);
    auto tokens = torch::empty({batch_size, 64, out_dim}, 
                               torch::dtype(torch::kFloat32).device(images.device()));
    
    dim3 grid(64, batch_size);
    patchify_and_project<<<grid, 64>>>(
        images.data_ptr<f32>(),
        proj_weights.data_ptr<f32>(),
        tokens.data_ptr<f32>(),
        batch_size,
        out_dim,
        static_cast<f32>(perturb_scale),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id),
        static_cast<f32>(pe_scale)
    );
    
    return tokens;
}

std::vector<torch::Tensor> compute_qkv_fn(
    torch::Tensor tokens,
    torch::Tensor QKV_weights,
    int64_t num_heads,
    int64_t head_dim,
    double perturb_scale,
    int64_t seed,
    int64_t layer_id
) {
    const i32 batch_size = tokens.size(0);
    const i32 num_tokens = tokens.size(1);
    const i32 in_dim = tokens.size(2);
    
    auto Q = torch::empty({batch_size, num_heads, num_tokens, head_dim}, 
                          torch::dtype(torch::kFloat32).device(tokens.device()));
    auto K = torch::empty_like(Q);
    auto V = torch::empty_like(Q);
    
    dim3 grid(num_tokens, num_heads, batch_size);
    compute_qkv<<<grid, 32>>>(
        tokens.data_ptr<f32>(),
        QKV_weights.data_ptr<f32>(),
        Q.data_ptr<f32>(),
        K.data_ptr<f32>(),
        V.data_ptr<f32>(),
        batch_size,
        num_tokens,
        in_dim,
        head_dim,
        num_heads,
        static_cast<f32>(perturb_scale),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
    );
    
    return {Q, K, V};
}

torch::Tensor attention_fn(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    const i32 batch_size = Q.size(0);
    const i32 num_heads = Q.size(1);
    const i32 num_tokens = Q.size(2);
    const i32 head_dim = Q.size(3);
    
    auto O = torch::empty_like(Q);
    
    dim3 grid(num_tokens, num_heads, batch_size);
    attention<<<grid, 32>>>(
        Q.data_ptr<f32>(),
        K.data_ptr<f32>(),
        V.data_ptr<f32>(),
        O.data_ptr<f32>(),
        batch_size,
        num_heads,
        num_tokens,
        head_dim
    );
    
    return O;
}

torch::Tensor project_and_residual_fn(
    torch::Tensor attn_output,
    torch::Tensor proj_weights,
    torch::Tensor residual,
    double perturb_scale,
    int64_t seed,
    int64_t layer_id
) {
    const i32 batch_size = attn_output.size(0);
    const i32 num_heads = attn_output.size(1);
    const i32 num_tokens = attn_output.size(2);
    const i32 head_dim = attn_output.size(3);
    const i32 out_dim = num_heads * head_dim;
    
    auto output = torch::empty({batch_size, num_tokens, out_dim}, 
                               torch::dtype(torch::kFloat32).device(attn_output.device()));
    
    dim3 grid(num_tokens, batch_size);
    project_and_residual<<<grid, 64>>>(
        attn_output.data_ptr<f32>(),
        proj_weights.data_ptr<f32>(),
        residual.data_ptr<f32>(),
        output.data_ptr<f32>(),
        batch_size,
        num_tokens,
        num_heads,
        head_dim,
        static_cast<f32>(perturb_scale),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
    );
    
    return output;
}

torch::Tensor final_projection_fn(
    torch::Tensor tokens,
    torch::Tensor proj_weights,
    double perturb_scale,
    int64_t seed,
    int64_t layer_id
) {
    const i32 batch_size = tokens.size(0);
    const i32 in_dim = tokens.size(2);
    const i32 out_dim = proj_weights.size(1);
    
    auto output = torch::empty({batch_size, 64, out_dim}, 
                               torch::dtype(torch::kFloat32).device(tokens.device()));
    
    dim3 grid(64, batch_size);
    final_projection<<<grid, 64>>>(
        tokens.data_ptr<f32>(),
        proj_weights.data_ptr<f32>(),
        output.data_ptr<f32>(),
        batch_size,
        in_dim,
        out_dim,
        static_cast<f32>(perturb_scale),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
    );
    
    return output;
}

torch::Tensor materialize_image_fn(
    torch::Tensor blend_weights,
    torch::Tensor learnable_patches,
    double perturb_scale,
    int64_t seed,
    int64_t layer_id
) {
    const i32 batch_size = blend_weights.size(0);
    const i32 num_patches = learnable_patches.size(0);
    
    auto output = torch::empty({batch_size, 64, 64, 3}, 
                               torch::dtype(torch::kFloat32).device(blend_weights.device()));
    
    dim3 grid(64, batch_size);
    materialize_image<<<grid, 64>>>(
        blend_weights.data_ptr<f32>(),
        learnable_patches.data_ptr<f32>(),
        output.data_ptr<f32>(),
        batch_size,
        num_patches,
        static_cast<f32>(perturb_scale),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id)
    );
    
    return output;
}

torch::Tensor compute_mse_scores_fn(
    torch::Tensor output,
    torch::Tensor ground_truth
) {
    const i32 batch_size = output.size(0);
    auto scores = torch::empty({batch_size}, 
                               torch::dtype(torch::kFloat32).device(output.device()));
    
    compute_mse_scores<<<batch_size, 256>>>(
        output.data_ptr<f32>(),
        ground_truth.data_ptr<f32>(),
        scores.data_ptr<f32>(),
        batch_size
    );
    
    return scores;
}

torch::Tensor compute_softmax_fn(
    torch::Tensor scores,
    double temperature
) {
    const i32 batch_size = scores.size(0);
    auto softmax_out = torch::empty({batch_size}, 
                                    torch::dtype(torch::kFloat32).device(scores.device()));
    
    compute_softmax<<<1, 256>>>(
        scores.data_ptr<f32>(),
        softmax_out.data_ptr<f32>(),
        batch_size,
        static_cast<f32>(temperature)
    );
    
    return softmax_out;
}

void accumulate_perturbation_fn(
    torch::Tensor weights,
    torch::Tensor softmax_weights,
    double perturb_scale,
    int64_t seed,
    int64_t layer_id,
    int64_t norm_size
) {
    const i32 batch_size = softmax_weights.size(0);
    const i32 total = weights.numel();
    const i32 N = weights.size(-1);
    const i32 M = total / N;
    
    const i32 threads = 256;
    const i32 blocks = (total + threads - 1) / threads;
    
    accumulate_perturbation_2d<<<blocks, threads>>>(
        weights.data_ptr<f32>(),
        softmax_weights.data_ptr<f32>(),
        batch_size,
        M,
        N,
        static_cast<f32>(perturb_scale),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id),
        static_cast<i32>(norm_size)
    );
}
;//
"""

cpp_source = """
//js
torch::Tensor patchify_fn(torch::Tensor images, torch::Tensor proj_weights, int64_t batch_size, double perturb_scale, int64_t seed, int64_t layer_id, double pe_scale);
std::vector<torch::Tensor> compute_qkv_fn(torch::Tensor tokens, torch::Tensor QKV_weights, int64_t num_heads, int64_t head_dim, double perturb_scale, int64_t seed, int64_t layer_id);
torch::Tensor attention_fn(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
torch::Tensor project_and_residual_fn(torch::Tensor attn_output, torch::Tensor proj_weights, torch::Tensor residual, double perturb_scale, int64_t seed, int64_t layer_id);
torch::Tensor final_projection_fn(torch::Tensor tokens, torch::Tensor proj_weights, double perturb_scale, int64_t seed, int64_t layer_id);
torch::Tensor materialize_image_fn(torch::Tensor blend_weights, torch::Tensor learnable_patches, double perturb_scale, int64_t seed, int64_t layer_id);
torch::Tensor compute_mse_scores_fn(torch::Tensor output, torch::Tensor ground_truth);
torch::Tensor compute_softmax_fn(torch::Tensor scores, double temperature);
void accumulate_perturbation_fn(torch::Tensor weights, torch::Tensor softmax_weights, double perturb_scale, int64_t seed, int64_t layer_id, int64_t norm_size);
;//
"""

module = load_inline(
    name="vit_evolutionary_v2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=[
        "patchify_fn", "compute_qkv_fn", "attention_fn", "project_and_residual_fn",
        "final_projection_fn", "materialize_image_fn",
        "compute_mse_scores_fn", "compute_softmax_fn", "accumulate_perturbation_fn"
    ],
    verbose=True,
    extra_cuda_cflags=["-O3", "-arch=sm_80"]
)

# ============ ViT Model ============
class ViTModel:
    def __init__(self, num_heads=4, num_layers=2, head_dim=16, num_patches=64, device="cuda"):
        self.device = device
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.num_patches = num_patches
        self.hidden_dim = num_heads * head_dim
        
        # Patchifier: [192, hidden_dim]
        self.patchifier = torch.randn(192, self.hidden_dim, device=device) / math.sqrt(192)
        
        # QKV weights per layer: [num_heads, 3, hidden_dim, head_dim]
        self.qkv_weights = []
        self.proj_weights = []
        for _ in range(num_layers):
            qkv = torch.randn(num_heads, 3, self.hidden_dim, head_dim, device=device) / math.sqrt(self.hidden_dim)
            self.qkv_weights.append(qkv)
            proj = torch.randn(self.hidden_dim, self.hidden_dim, device=device) / math.sqrt(self.hidden_dim)
            self.proj_weights.append(proj)
        
        # Final projection: [hidden_dim, num_patches]
        self.final_proj = torch.randn(self.hidden_dim, num_patches, device=device) / math.sqrt(self.hidden_dim)
        
        # Learnable patches: [num_patches, 192]
        self.learnable_patches = torch.randn(num_patches, 192, device=device) * 0.5
        
        # Layer IDs for perturbations
        self.layer_ids = {
            'patchifier': 0,
            'final_proj': 1,
            'patches': 2,
        }
        for i in range(num_layers):
            self.layer_ids[f'qkv_{i}'] = 100 + i * 10
            self.layer_ids[f'proj_{i}'] = 100 + i * 10 + 1
    
    def forward(self, images, batch_size, seed, perturb_scale, pe_scale):
        # Patchify
        x = module.patchify_fn(
            images, self.patchifier, batch_size, 
            perturb_scale, seed, self.layer_ids['patchifier'], pe_scale
        )
        
        # Attention layers
        for i in range(self.num_layers):
            Q, K, V = module.compute_qkv_fn(
                x, self.qkv_weights[i], self.num_heads, self.head_dim,
                perturb_scale, seed, self.layer_ids[f'qkv_{i}']
            )
            
            attn_out = module.attention_fn(Q, K, V)
            
            x = module.project_and_residual_fn(
                attn_out, self.proj_weights[i], x,
                perturb_scale, seed, self.layer_ids[f'proj_{i}']
            )
        
        # Final projection to blend weights
        blend_weights = module.final_projection_fn(
            x, self.final_proj, perturb_scale, seed, self.layer_ids['final_proj']
        )
        
        # Materialize image
        output = module.materialize_image_fn(
            blend_weights, self.learnable_patches,
            perturb_scale, seed, self.layer_ids['patches']
        )
        
        return output
    
    def accumulate(self, softmax_weights, seed, perturb_scale):
        # Patchifier
        module.accumulate_perturbation_fn(
            self.patchifier, softmax_weights, perturb_scale, seed, 
            self.layer_ids['patchifier'], 192
        )
        
        # QKV and proj weights
        for i in range(self.num_layers):
            module.accumulate_perturbation_fn(
                self.qkv_weights[i], softmax_weights, perturb_scale, seed,
                self.layer_ids[f'qkv_{i}'], self.hidden_dim
            )
            module.accumulate_perturbation_fn(
                self.proj_weights[i], softmax_weights, perturb_scale, seed,
                self.layer_ids[f'proj_{i}'], self.hidden_dim
            )
        
        # Final projection
        module.accumulate_perturbation_fn(
            self.final_proj, softmax_weights, perturb_scale, seed,
            self.layer_ids['final_proj'], self.hidden_dim
        )
        
        # Learnable patches
        module.accumulate_perturbation_fn(
            self.learnable_patches, softmax_weights, perturb_scale, seed,
            self.layer_ids['patches'], 8
        )

def train_step(model, images, ground_truth, batch_size, seed, perturb_scale, pe_scale, temperature):
    output = model.forward(images, batch_size, seed, perturb_scale, pe_scale)
    scores = module.compute_mse_scores_fn(output, ground_truth)
    softmax_weights = module.compute_softmax_fn(scores, temperature)
    model.accumulate(softmax_weights, seed, perturb_scale)
    return output, scores, softmax_weights

# ============ Test ============
print("=" * 60)
print("Evolutionary ViT Training")
print("=" * 60)

batch_size = 256
num_heads = 4
num_layers = 2
head_dim = 16
num_patches = 64
perturb_scale = 0.1
pe_scale = 1.0
temperature = 0.00001

model = ViTModel(num_heads, num_layers, head_dim, num_patches)

# Load ground truth
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
        image = Image.open(img_path)
        return self.transform(image)

dataset = SimpleDataloader("data\\MNIST\\dataset-part1")
ground_truth = dataset.get_image().to("cuda").permute(1, 2, 0).contiguous()
print(f"Ground truth range: [{ground_truth.min():.3f}, {ground_truth.max():.3f}]")

print(f"\nConfig:")
print(f"  Batch size: {batch_size}")
print(f"  Num heads: {num_heads}, Num layers: {num_layers}, Head dim: {head_dim}")
print(f"  Perturb scale: {perturb_scale}")
print(f"  Temperature: {temperature}")

num_epochs = 1 << 14
losses = []

for epoch in range(num_epochs):
    seed = epoch * 1000
    
    perturb_scale *= 0.99998
    temperature *= 0.9999
    
    # Input images (could be noise or the target itself)
    images = ground_truth.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
    noise_sigma = 0.01
    images = images * (1.0 - noise_sigma) + torch.randn_like(images) * noise_sigma

    output, scores, softmax_weights = train_step(
        model, images, ground_truth, batch_size, seed, perturb_scale, pe_scale, temperature
    )
    
    scores_np = scores.cpu().numpy()
    softmax_np = softmax_weights.cpu().numpy()
    
    best_mse = -scores_np.max()
    mean_mse = -scores_np.mean()
    losses.append(best_mse)
    
    entropy = -(softmax_np * np.log(softmax_np + 1e-10)).sum()
    
    if epoch % 100 == 0 or epoch == num_epochs - 1:
        try:
            dds = dds_from_tensor(output.permute(0, 3, 1, 2)[0:1])
            dds.save(".tmp/vit_evo_result.dds")
        except Exception as e:
            print(f"Could not save: {e}")
        
        print(f"\nEpoch {epoch + 1}:")
        print(f"  Best MSE: {best_mse:.6f}, Mean MSE: {mean_mse:.6f}")
        print(f"  Softmax entropy: {entropy:.2f} (max={np.log(batch_size):.2f})")
        print(f"  Perturb scale: {perturb_scale:.6f}, Temperature: {temperature:.8f}")
        print(f"  Top 3 weights: {np.sort(softmax_np)[-3:][::-1].round(4)}")

print(f"\n" + "=" * 60)
print(f"Loss progression: {losses[0]:.6f} -> {losses[-1]:.6f}")
if losses[0] > 0:
    print(f"Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
print("=" * 60)
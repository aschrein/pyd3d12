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

// ============ Forward Pass: Materialize Image ============

__global__ void materialize_image(
    const i8*  __restrict__ blend_weights,      // [64, 64] int8
    const i8*  __restrict__ learnable_patches,  // [64, 192] int8
          i8*  __restrict__ output,             // [batch, 64, 64, 3] int8
    const i32              batch_size,
    const f32              weights_scale,
    const f32              patches_scale,
    const f32              perturb_scale,
    const u32              base_seed
) {
    const i32 pos_idx   = blockIdx.x;  // 0-63
    const i32 batch_idx = blockIdx.y;
    const i32 tid       = threadIdx.x;
    
    const i32 py = pos_idx / 8;
    const i32 px = pos_idx % 8;
    
    // Load blend weights with perturbation, apply tanh
    __shared__ f32 weights[64];
    for (i32 i = tid; i < 64; i += 32) {
        f32 base_w = static_cast<f32>(blend_weights[pos_idx * 64 + i]) * weights_scale;
        f32 perturb = get_perturbation(base_seed, batch_idx, 0, pos_idx, i, perturb_scale, 64);
        weights[i] = tanhf(base_w + perturb);
    }
    __syncthreads();
    
    // Blend patches for each pixel
    for (i32 pixel_idx = tid; pixel_idx < 192; pixel_idx += 32) {
        const i32 local_y = pixel_idx / 24;
        const i32 local_x = (pixel_idx % 24) / 3;
        const i32 channel = pixel_idx % 3;
        
        f32 blended = 0.0f;
        for (i32 p = 0; p < 64; p++) {
            f32 patch_pixel = static_cast<f32>(learnable_patches[p * 192 + pixel_idx]) * patches_scale;
            patch_pixel += get_perturbation(base_seed, batch_idx, 1, p, pixel_idx, perturb_scale, 8);
            blended += patch_pixel * weights[p];
        }
        
        const i32 img_y = py * 8 + local_y;
        const i32 img_x = px * 8 + local_x;
        
        // Clamp and store as int8
        output[batch_idx * 64 * 64 * 3 + img_y * 64 * 3 + img_x * 3 + channel] = 
            static_cast<i8>(min(max(rintf(blended / patches_scale), -127.0f), 127.0f));
    }
}

// ============ Compute MSE Scores ============

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
        f32 diff = static_cast<f32>(output[batch_idx * num_pixels + i] - ground_truth[i]) / 127.0f;
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

// ============ Compute Softmax ============

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

__global__ void accumulate_perturbation(
    const i8*  __restrict__ base_weights,
          i8*  __restrict__ out_weights,
    const f32* __restrict__ softmax_weights,
    const i32 batch_size,
    const i32 M,
    const i32 N,
    const f32 scale,
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
    
    f32 base_val = static_cast<f32>(base_weights[idx]) * scale;
    
    f32 perturb_sum = 0.0f;
    for (i32 b = 0; b < batch_size; b++) {
        f32 p = get_perturbation(base_seed, b, layer_id, row, col, perturb_scale, norm_size);
        perturb_sum += sqrtf(softmax_weights[b]) * p;
    }
    
    f32 new_val = base_val + perturb_sum;
    out_weights[idx] = static_cast<i8>(min(max(rintf(new_val / scale), -127.0f), 127.0f));
}

// ============ Host Functions ============

torch::Tensor materialize_image_fn(
    torch::Tensor blend_weights,
    torch::Tensor learnable_patches,
    int64_t batch_size,
    double weights_scale,
    double patches_scale,
    double perturb_scale,
    int64_t seed
) {
    auto output = torch::empty({batch_size, 64, 64, 3}, 
                               torch::dtype(torch::kInt8).device(blend_weights.device()));
    
    dim3 grid(64, batch_size);
    materialize_image<<<grid, 32>>>(
        blend_weights.data_ptr<i8>(),
        learnable_patches.data_ptr<i8>(),
        output.data_ptr<i8>(),
        static_cast<i32>(batch_size),
        static_cast<f32>(weights_scale),
        static_cast<f32>(patches_scale),
        static_cast<f32>(perturb_scale),
        static_cast<u32>(seed)
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
        output.data_ptr<i8>(),
        ground_truth.data_ptr<i8>(),
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

torch::Tensor accumulate_fn(
    torch::Tensor weights,
    torch::Tensor softmax_weights,
    double scale,
    double perturb_scale,
    int64_t seed,
    int64_t layer_id,
    int64_t norm_size
) {
    const i32 batch_size = softmax_weights.size(0);
    const i32 M = weights.size(0);
    const i32 N = weights.size(1);
    const i32 total = M * N;
    
    auto out_weights = torch::empty_like(weights);
    
    const i32 threads = 256;
    const i32 blocks = (total + threads - 1) / threads;
    
    accumulate_perturbation<<<blocks, threads>>>(
        weights.data_ptr<i8>(),
        out_weights.data_ptr<i8>(),
        softmax_weights.data_ptr<f32>(),
        batch_size, M, N,
        static_cast<f32>(scale),
        static_cast<f32>(perturb_scale),
        static_cast<u32>(seed),
        static_cast<u32>(layer_id),
        static_cast<i32>(norm_size)
    );
    
    return out_weights;
}
;//
"""

cpp_source = """
torch::Tensor materialize_image_fn(torch::Tensor blend_weights, torch::Tensor learnable_patches, int64_t batch_size, double weights_scale, double patches_scale, double perturb_scale, int64_t seed);
torch::Tensor compute_mse_scores_fn(torch::Tensor output, torch::Tensor ground_truth);
torch::Tensor compute_softmax_fn(torch::Tensor scores, double temperature);
torch::Tensor accumulate_fn(torch::Tensor weights, torch::Tensor softmax_weights, double scale, double perturb_scale, int64_t seed, int64_t layer_id, int64_t norm_size);
"""

module = load_inline(
    name="simple_int8_v1",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=[
        "materialize_image_fn", "compute_mse_scores_fn", "compute_softmax_fn", "accumulate_fn"
    ],
    verbose=True,
    extra_cuda_cflags=["-O3", "-arch=sm_80"]
)

# ============ Simple Model ============
class SimpleBlendModel:
    def __init__(self, weights_scale=6.0, patches_scale=6.0, device="cuda"):
        self.device = device
        self.weights_scale = weights_scale / 127.0
        self.patches_scale = patches_scale / 127.0
        
        # Blend weights: [64 positions, 64 patches] - zero init
        self.blend_weights = torch.zeros(64, 64, dtype=torch.int8, device=device)
        
        # Learnable patches: [64 patches, 192 pixels]
        patches_f = torch.randn(64, 192, device=device) * 0.5
        self.learnable_patches = (patches_f / self.patches_scale).round().clamp(-127, 127).to(torch.int8)
    
    def forward(self, batch_size, seed, perturb_scale):
        return module.materialize_image_fn(
            self.blend_weights, self.learnable_patches,
            batch_size, self.weights_scale, self.patches_scale,
            perturb_scale, seed
        )
    
    def accumulate(self, softmax_weights, seed, perturb_scale):
        self.blend_weights = module.accumulate_fn(
            self.blend_weights, softmax_weights,
            self.weights_scale, perturb_scale, seed, 0, 64
        )
        self.learnable_patches = module.accumulate_fn(
            self.learnable_patches, softmax_weights,
            self.patches_scale, perturb_scale, seed, 1, 8
        )

# ============ Training ============
print("=" * 60)
print("Simple Evolutionary Patch Blending (INT8)")
print("=" * 60)

batch_size    = 256
perturb_scale = 0.2
temperature   = 0.00001

model = SimpleBlendModel()

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
ground_truth = (ground_truth_f * 127.0).round().clamp(-127, 127).to(torch.int8)

print(f"Ground truth: float [{ground_truth_f.min():.3f}, {ground_truth_f.max():.3f}] -> int8 [{ground_truth.min()}, {ground_truth.max()}]")
print(f"\nConfig: batch={batch_size}, perturb={perturb_scale}, temp={temperature}")

num_epochs = 1 << 14
losses = []

for epoch in range(num_epochs):
    seed = epoch * 1000
    perturb_scale *= 0.99998
    temperature *= 0.9999

    output = model.forward(batch_size, seed, perturb_scale)
    scores = module.compute_mse_scores_fn(output, ground_truth)
    softmax_weights = module.compute_softmax_fn(scores, temperature)
    model.accumulate(softmax_weights, seed, perturb_scale)
    
    scores_np = scores.cpu().numpy()
    softmax_np = softmax_weights.cpu().numpy()
    best_mse = -scores_np.max()
    losses.append(best_mse)
    
    if epoch % 100 == 0 or epoch == num_epochs - 1:
        entropy = -(softmax_np * np.log(softmax_np + 1e-10)).sum()
        try:
            dds = dds_from_tensor(output[0:1].float().permute(0, 3, 1, 2) / 127.0)
            dds.save(".tmp/epoch.dds")
        except: pass
        
        print(f"\nEpoch {epoch+1}: MSE={best_mse:.6f}, entropy={entropy:.2f}, "
              f"perturb={perturb_scale:.6f}, temp={temperature:.8f}")
        print(f"  weights=[{model.blend_weights.min()}, {model.blend_weights.max()}], "
              f"patches=[{model.learnable_patches.min()}, {model.learnable_patches.max()}]")
        print(f"  top3: {np.sort(softmax_np)[-3:][::-1].round(4)}")

print(f"\n{'='*60}")
print(f"Loss: {losses[0]:.6f} -> {losses[-1]:.6f} ({(losses[0]-losses[-1])/losses[0]*100:.1f}% improvement)")
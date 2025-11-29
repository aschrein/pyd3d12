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
    u32 base_seed,
    i32 batch_idx,
    u32 layer_id,
    i32 row,
    i32 col,
    f32 scale,
    i32 normalization_size
) {
    const f32 norm = sqrtf((f32)normalization_size);
    // Low-rank perturbation: u_row * v_col
    u32 u_seed = base_seed ^ (batch_idx * 12345) ^ (layer_id * 111111) ^ (row * 67890);
    u32 v_seed = base_seed ^ (batch_idx * 12345) ^ (layer_id * 111111) ^ ((col + 100000) * 67890);
    
    f32 u_val = random_normal(u_seed) * scale;
    f32 v_val = random_normal(v_seed) * scale;
    
    return u_val * v_val / norm;
}

// ============ Forward Pass: Materialize Image ============
// For each position in 8x8 grid, blend 64 learnable patches using weights

__global__ void materialize_image(
    const f32* __restrict__ blend_weights,      // [64, 64] - for each position, weights over 64 patches
    const f32* __restrict__ learnable_patches,  // [64, 8, 8, 3] = [64, 192] flattened
          f32* __restrict__ output,             // [batch, 64, 64, 3]
    const i32              batch_size,
    const f32              perturb_scale,
    const u32              base_seed,
    const u32              weights_layer_id,
    const u32              patches_layer_id
) {
    const i32 pos_idx   = blockIdx.x;  // 0-63 (which position in 8x8 grid)
    const i32 batch_idx = blockIdx.y;
    const i32 tid       = threadIdx.x;
    
    const i32 py = pos_idx / 8;  // Position Y in grid (0-7)
    const i32 px = pos_idx % 8;  // Position X in grid (0-7)
    
    // Load blend weights for this position with perturbation
    __shared__ f32 weights[64];
    for (i32 i = tid; i < 64; i += 32) {
        f32 base_w = blend_weights[pos_idx * 64 + i];
        f32 perturb = get_perturbation(base_seed, batch_idx, weights_layer_id, pos_idx, i, perturb_scale, /* normalization_size */64);
        weights[i] = tanhf(base_w + perturb);
    }
    __syncthreads();
    
    
    // Blend patches for each pixel in 8x8x3 = 192 values
    for (i32 pixel_idx = tid; pixel_idx < 192; pixel_idx += 32) {
        const i32 local_y = pixel_idx / 24;        // 0-7
        const i32 local_x = (pixel_idx % 24) / 3;  // 0-7
        const i32 channel = pixel_idx % 3;         // 0-2
        
        f32 blended = 0.0f;
        for (i32 p = 0; p < 64; p++) {
            f32 patch_pixel = learnable_patches[p * 192 + pixel_idx];
            // Add perturbation to patch pixels
            patch_pixel += get_perturbation(base_seed, batch_idx, patches_layer_id, p, pixel_idx, perturb_scale, /* normalization_size */ 8);
            blended += patch_pixel * weights[p];
        }
        
        // Global image coordinates
        const i32 img_y = py * 8 + local_y;
        const i32 img_x = px * 8 + local_x;
        output[batch_idx * 64 * 64 * 3 + img_y * 64 * 3 + img_x * 3 + channel] = blended;
    }
}

// ============ Compute MSE Scores ============

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
        f32 diff = output[batch_idx * num_pixels + i] - ground_truth[i];  // ground_truth is not batched
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
        // Negative MSE so higher is better for softmax
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

__global__ void accumulate_perturbation_2d(
          f32* __restrict__ weights,       // [M, N] - updated in place
    const f32* __restrict__ softmax_weights,
    const i32 batch_size,
    const i32 M,
    const i32 N,
    const f32 perturb_scale,
    const u32 base_seed,
    const u32 layer_id,
    const i32 normalization_size // Size to normalize the dot product to be unit variance
) {
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    const i32 total = M * N;
    
    if (idx >= total) return;
    
    const i32 row = idx / N;
    const i32 col = idx % N;
    
    f32 perturb_sum = 0.0f;
    for (i32 b = 0; b < batch_size; b++) {
        f32 p = get_perturbation(base_seed, b, layer_id, row, col, perturb_scale, normalization_size);
        perturb_sum += sqrtf(softmax_weights[b]) * p;
    }
    
    weights[idx] += perturb_sum;
}

// ============ Host Functions ============

torch::Tensor materialize_image_fn(
    torch::Tensor blend_weights,
    torch::Tensor learnable_patches,
    int64_t batch_size,
    double perturb_scale,
    int64_t seed
) {
    auto output = torch::empty({batch_size, 64, 64, 3}, 
                               torch::dtype(torch::kFloat32).device(blend_weights.device()));
    
    dim3 grid(64, batch_size);
    
    materialize_image<<<grid, 32>>>(
        blend_weights.data_ptr<f32>(),
        learnable_patches.data_ptr<f32>(),
        output.data_ptr<f32>(),
        static_cast<i32>(batch_size),
        static_cast<f32>(perturb_scale),
        static_cast<u32>(seed),
        0,  // weights_layer_id
        1   // patches_layer_id
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

void accumulate_blend_weights_fn(
    torch::Tensor blend_weights,
    torch::Tensor softmax_weights,
    double perturb_scale,
    int64_t seed,
    int64_t normalization_size
) {
    const i32 batch_size = softmax_weights.size(0);
    const i32 M = blend_weights.size(0);
    const i32 N = blend_weights.size(1);
    const i32 total = M * N;
    
    const i32 threads = 256;
    const i32 blocks = (total + threads - 1) / threads;
    
    accumulate_perturbation_2d<<<blocks, threads>>>(
        blend_weights.data_ptr<f32>(),
        softmax_weights.data_ptr<f32>(),
        batch_size,
        M,
        N,
        static_cast<f32>(perturb_scale),
        static_cast<u32>(seed),
        0,  // weights_layer_id
        static_cast<i32>(normalization_size)
    );
}

void accumulate_patches_fn(
    torch::Tensor learnable_patches,
    torch::Tensor softmax_weights,
    double perturb_scale,
    int64_t seed,
    int64_t normalization_size
) {
    const i32 batch_size = softmax_weights.size(0);
    const i32 M = 64;   // num patches
    const i32 N = 192;  // pixels per patch
    const i32 total = M * N;
    
    const i32 threads = 256;
    const i32 blocks = (total + threads - 1) / threads;
    
    accumulate_perturbation_2d<<<blocks, threads>>>(
        learnable_patches.data_ptr<f32>(),
        softmax_weights.data_ptr<f32>(),
        batch_size,
        M,
        N,
        static_cast<f32>(perturb_scale),
        static_cast<u32>(seed),
        1,  // patches_layer_id
        static_cast<i32>(normalization_size)
    );
}
;//
"""

cpp_source = """
torch::Tensor materialize_image_fn(torch::Tensor blend_weights, torch::Tensor learnable_patches, int64_t batch_size, double perturb_scale, int64_t seed);
torch::Tensor compute_mse_scores_fn(torch::Tensor output, torch::Tensor ground_truth);
torch::Tensor compute_softmax_fn(torch::Tensor scores, double temperature);
void accumulate_blend_weights_fn(torch::Tensor blend_weights, torch::Tensor softmax_weights, double perturb_scale, int64_t seed, int64_t normalization_size);
void accumulate_patches_fn(torch::Tensor learnable_patches, torch::Tensor softmax_weights, double perturb_scale, int64_t seed, int64_t normalization_size);
"""

module = load_inline(
    name="simple_evolutionary_v1",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=[
        "materialize_image_fn", "compute_mse_scores_fn", "compute_softmax_fn",
        "accumulate_blend_weights_fn", "accumulate_patches_fn"
    ],
    verbose=True,
    extra_cuda_cflags=["-O3", "-arch=sm_80"]
)

# ============ Simple Model ============
class SimpleBlendModel:
    def __init__(self, device="cuda"):
        self.device = device
        
        # Blend weights: [64 positions, 64 patches] - logits for softmax
        self.blend_weights = torch.zeros(64, 64, device=device)
        
        # Learnable patches: [64 patches, 192 pixels (8x8x3)]
        # Initialize with random colors
        self.learnable_patches = torch.randn(64, 192, device=device) * 0.5
    
    def forward(self, batch_size, seed, perturb_scale):
        return module.materialize_image_fn(
            self.blend_weights,
            self.learnable_patches,
            batch_size,
            perturb_scale,
            seed
        )
    
    def accumulate(self, softmax_weights, seed, perturb_scale):
        module.accumulate_blend_weights_fn(
            self.blend_weights, softmax_weights, perturb_scale, seed, 64
        )
        module.accumulate_patches_fn(
            self.learnable_patches, softmax_weights, perturb_scale, seed, 8
        )

def train_step(model, ground_truth, batch_size, seed, perturb_scale, temperature):
    # Forward pass with perturbations
    output = model.forward(batch_size, seed, perturb_scale)
    
    # Compute scores (negative MSE)
    scores = module.compute_mse_scores_fn(output, ground_truth)
    
    # Softmax to get weights
    softmax_weights = module.compute_softmax_fn(scores, temperature)
    
    # Accumulate weighted perturbations
    model.accumulate(softmax_weights, seed, perturb_scale)
    
    return output, scores, softmax_weights

# ============ Test ============
print("=" * 60)
print("Simple Evolutionary Patch Blending")
print("=" * 60)

batch_size    = 256
perturb_scale = 0.1
temperature   = 0.00001

model = SimpleBlendModel()

# Or load from dataset
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
ground_truth = dataset.get_image().to("cuda").permute(1, 2, 0).contiguous()  # [64, 64, 3]
print("Loaded image from dataset")

print(f"Ground truth range: [{ground_truth.min():.3f}, {ground_truth.max():.3f}]")

print(f"\nConfig:")
print(f"  Batch size: {batch_size}")
print(f"  Perturb scale: {perturb_scale}")
print(f"  Temperature: {temperature}")

num_epochs = 1 << 14
losses = []

for epoch in range(num_epochs):
    seed = epoch * 1000

    t = (epoch - num_epochs) / num_epochs
    perturb_scale = perturb_scale * 0.99998
    temperature = temperature * 0.9999

    output, scores, softmax_weights = train_step(
        model, ground_truth, batch_size, seed, perturb_scale, temperature
    )
    
    scores_np = scores.cpu().numpy()
    softmax_np = softmax_weights.cpu().numpy()
    
    best_mse = -scores_np.max()
    mean_mse = -scores_np.mean()
    losses.append(best_mse)
    
    entropy = -(softmax_np * np.log(softmax_np + 1e-10)).sum()
    
    if epoch % 100 == 0 or epoch == num_epochs - 1:
        try:
            dds = dds_from_tensor(output.permute(0, 3, 1, 2)[0:1, :, :, :])
            dds.save(".tmp/simple_evo_result.dds")
        except Exception as e:
            print(f"Could not save: {e}")
            
        print(f"\nEpoch {epoch + 1}:")
        print(f"  Best MSE: {best_mse:.6f}, Mean MSE: {mean_mse:.6f}")
        print(f"  Softmax entropy: {entropy:.2f} (max={np.log(batch_size):.2f})")
        print(f"  Perturb scale: {perturb_scale:.6f}, Temperature: {temperature:.6f}")
        print(f"  Top 3 weights: {np.sort(softmax_np)[-3:][::-1].round(4)}")
        print(f"  Output range: [{output[0].min():.3f}, {output[0].max():.3f}]")

print(f"\n" + "=" * 60)
print(f"Loss progression: {losses[0]:.6f} -> {losses[-1]:.6f}")
if losses[0] > 0:
    print(f"Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
print("=" * 60)

# Save final output
try:
    final_output = model.forward(1, 999999, 0.0)  # No perturbation for final
    dds = dds_from_tensor(final_output.permute(0, 3, 1, 2))
    dds.save(".tmp/simple_evo_result.dds")
    print("Saved result to .tmp/simple_evo_result.dds")
except Exception as e:
    print(f"Could not save: {e}")
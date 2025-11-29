# MIT License
# Copyright (c) 2025 Anton Schreiner

import torch
from torch.utils.cpp_extension import load_inline

# CUDA kernel as a string
cuda_source = """
//js

#define f32 float
#define i32 int32_t
#define u32 uint32_t
#define u64 uint64_t
#define ifor(N) for(i32 i=0;i<(N);i++)
#define jfor(N) for(i32 j=0;j<(N);j++)
#define kfor(N) for(i32 k=0;k<(N);k++)
#define xfor(N) for(i32 x=0;x<(N);x++)
#define yfor(N) for(i32 y=0;y<(N);y++)

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

struct f32x2 {
    f32 x;
    f32 y;
};

static __device__ f32x2 make_f32x2(f32 x, f32 y) {
    f32x2 result;
    result.x = x;
    result.y = y;
    return result;
}


// https://github.com/skeeto/hash-prospector      
static __device__ u32 lowbias32(u32 x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

static __device__ f32x2 random_f32x2(u32 *seed) {
    *seed = lowbias32(*seed);
    return make_f32x2((f32)(*seed & 0xFFFF) / 0xFFFF, (f32)((*seed >> 16) & 0xFFFF) / 0xFFFF);
}

static __device__ f32x2 box_muller_transform(f32x2 uv) {
    const f32 radius    = sqrtf(-2.0f * logf(uv.x));
    const f32 theta     = 2.0f * M_PI * uv.y;
    const f32x2 result  = make_f32x2(radius * cosf(theta), radius * sinf(theta));
    return result;
}

__global__ void box_muller_gaussian_kernel(
    f32* __restrict__ output,
    const i32 total_elements,
    const u32 seed
) {
    const i32 idx    = blockIdx.x * blockDim.x + threadIdx.x;
    const i32 stride = blockDim.x * gridDim.x;
    u32 state        = seed + idx;
    
    // Process pairs of elements for Box-Muller
    const f32x2 result      = box_muller_transform(random_f32x2(&state));
    const f32 z0            = result.x;
    const f32 z1            = result.y;
    output[idx * 2 + 0]     = z0;
    output[idx * 2 + 1]     = z1;
}

torch::Tensor fill_gaussian_box_muller(torch::Tensor output, int64_t seed) {
    // Ensure tensor is contiguous and on CUDA
    TORCH_CHECK(output.is_cuda(), "Tensor must be on CUDA device");
    TORCH_CHECK(output.is_contiguous(), "Tensor must be contiguous");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "Tensor must be float32");
    
    const i32 total_elements = output.numel();
    f32* data_ptr            = output.data_ptr<f32>();
    
    // Launch configuration
    const i32 threads = 256;
    const i32 blocks = min((total_elements / 2 + threads - 1) / threads, (i32)65535);
    
    box_muller_gaussian_kernel<<<blocks, threads>>>(
        data_ptr,
        total_elements,
        static_cast<u64>(seed)
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}
;//
"""

cpp_source = """
torch::Tensor fill_gaussian_box_muller(torch::Tensor output, int64_t seed);
"""

# Compile the extension
gaussian_module = load_inline(
    name="gaussian_box_muller",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fill_gaussian_box_muller"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


def fill_tensor_gaussian(B: int, C: int, H: int, W: int, seed: int, device: str = "cuda") -> torch.Tensor:
    """Create a B x C x H x W tensor filled with Gaussian random values."""
    # Allocate contiguous tensor
    tensor = torch.empty(B, C, H, W, dtype=torch.float32, device=device)
    
    # Ensure contiguity (already guaranteed by torch.empty, but explicit check)
    assert tensor.is_contiguous(), "Tensor must be contiguous"
    
    # Fill with Gaussian values using our CUDA kernel
    gaussian_module.fill_gaussian_box_muller(tensor, seed)
    
    return tensor


if __name__ == "__main__":
    # Example usage
    B, C, H, W = 2, 3, 64, 64
    seed = 42
    
    tensor = fill_tensor_gaussian(B, C, H, W, seed)
    
    print(f"Shape: {tensor.shape}")
    print(f"Device: {tensor.device}")
    print(f"Contiguous: {tensor.is_contiguous()}")
    print(f"Mean: {tensor.mean().item():.4f} (expected ~0)")
    print(f"Std: {tensor.std().item():.4f} (expected ~1)")
    print(f"Min: {tensor.min().item():.4f}")
    print(f"Max: {tensor.max().item():.4f}")
    
    # Verify reproducibility
    tensor2 = fill_tensor_gaussian(B, C, H, W, seed)
    print(f"Same seed produces same result: {torch.allclose(tensor, tensor2)}")
    
    tensor3 = fill_tensor_gaussian(B, C, H, W, seed + 1)
    print(f"Different seed produces different result: {not torch.allclose(tensor, tensor3)}")


import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
//js
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// Tensor Core matrix multiply: C = A * B
// Using wmma with 16x16x16 tiles in half precision
__global__ void tensor_core_matmul_16x16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C
) {
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    
    // Initialize output to zero
    wmma::fill_fragment(c_frag, __float2half(0.0f));
    
    // Load matrices into fragments
    // Leading dimension is 16 for both (row stride)
    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);
    
    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the result
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

torch::Tensor matmul_tensor_core(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(A.scalar_type() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.scalar_type() == torch::kFloat16, "B must be float16");
    TORCH_CHECK(A.size(0) == 16 && A.size(1) == 16, "A must be 16x16");
    TORCH_CHECK(B.size(0) == 16 && B.size(1) == 16, "B must be 16x16");
    
    // Output in float16 (accumulator precision)
    auto C = torch::zeros({16, 16}, torch::dtype(torch::kFloat16).device(A.device()));
    
    // Launch with exactly one warp (32 threads)
    // Tensor core operations are warp-cooperative
    tensor_core_matmul_16x16<<<1, 32>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>())
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return C;
}
;//
"""

cpp_source = """
torch::Tensor matmul_tensor_core(torch::Tensor A, torch::Tensor B);
"""

# Compile - requires compute capability 7.0+ (Volta, Turing, Ampere, etc.)
tensor_core_module = load_inline(
    name="tensor_core_matmul",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_tensor_core"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-arch=sm_80"]  # Adjust sm_XX for your GPU
)


if __name__ == "__main__":
    # Create test matrices in float16
    A = torch.randn(16, 16, dtype=torch.float16, device="cuda")
    B = torch.randn(16, 16, dtype=torch.float16, device="cuda")
    
    # Tensor core multiplication
    C_tc = tensor_core_module.matmul_tensor_core(A, B)
    
    # Reference using PyTorch (also uses tensor cores internally for fp16)
    C_ref = torch.mm(A.float(), B.float(), out_dtype=torch.float32).half()
    
     # Print results
    
    print(f"A shape: {A.shape}, dtype: {A.dtype}")
    print(f"B shape: {B.shape}, dtype: {B.dtype}")
    print(f"C shape: {C_tc.shape}, dtype: {C_tc.dtype}")
    print(f"\nMax absolute error vs reference: {(C_tc - C_ref).abs().max().item():.6e}")
    print(f"All close: {torch.allclose(C_tc, C_ref, atol=1e-3)}")
    
    print(f"\nC_tc[0, :4]: {C_tc[0, :4].tolist()}")
    print(f"C_ref[0, :4]: {C_ref[0, :4].tolist()}")

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
//js
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

__device__ __forceinline__ int8_t saturate_int8(int32_t val) {
    return static_cast<int8_t>(min(max(val, -128), 127));
}

__global__ void tensor_core_matmul_int8_saturate(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int8_t* __restrict__ C
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

    // Convert int32 -> int8 with saturation directly in registers
    // wmma::fragment<wmma::accumulator, 16, 16, 16, signed char> c_frag_i8;
    // for (int i = 0; i < c_frag.num_elements; i++) {
    //     int32_t val    = c_frag.x[i];
    //     c_frag_i8.x[i] = static_cast<signed char>(min(max(val, -128), 127));
    // }
    // // Store directly - layout matches between same-shaped accumulators
    // wmma::store_matrix_sync(C, c_frag_i8, 16, wmma::mem_row_major);
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
;//
"""

cpp_source = """
torch::Tensor matmul_int8_out(torch::Tensor A, torch::Tensor B);
"""

int8_module = load_inline(
    name="tensor_core_int8_saturate",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_int8_out"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-arch=sm_80"]
)


if __name__ == "__main__":
    # Small values to avoid saturation for testing
    A = torch.randint(-8, 8, (16, 16), dtype=torch.int8, device="cuda")
    B = torch.randint(-8, 8, (16, 16), dtype=torch.int8, device="cuda")
    
    C_tc = int8_module.matmul_int8_out(A, B)
    C_ref = torch.mm(A.to(torch.float32), B.to(torch.float32)).clamp(-128.0, 127.0).to(torch.int8)
    
    print(f"C dtype: {C_tc.dtype}")
    print(f"Exact match: {torch.equal(C_tc, C_ref)}")

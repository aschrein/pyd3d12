import math
import random
from matplotlib import image
import numpy as np
import ctypes
import pyopencl as cl
import os

os.environ["PYOPENCL_CTX"] = "0"

TERMINAL_COLOR_GREEN = "\033[32m"
TERMINAL_COLOR_RED   = "\033[31m"
TERMINAL_COLOR_RESET = "\033[0m"

DEBUG_CPU_EMULATION = 0

ctx   = cl.create_some_context()
queue = cl.CommandQueue(ctx)

def dims_get_total(dims):
    total = 1
    for d in dims:
        total *= d
    return total

free_tensor_pool = {}
allocated_tensor_pool = {}

def get_number_of_allocated_buffers():
    return sum(len(buffs) for buffs in allocated_tensor_pool.values())

def get_number_of_free_buffers():
    return sum(len(buffs) for buffs in free_tensor_pool.values())

def make_array(shape):
    size = dims_get_total(shape) * 4
    # if size in free_tensor_pool and len(free_tensor_pool[size]) > 0:
    #     # print("Reusing buffer of size:", size)
    #     return free_tensor_pool[size].pop()
    buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=size)
    # allocated_tensor_pool[size] = allocated_tensor_pool.get(size, []) + [buf]
    return buf

def free_all_buffers():
    for size, buffs in allocated_tensor_pool.items():
        for buf in buffs:
            free_buffer(buf, size)

def free_buffer(buffer, size_in_bytes):
    if size_in_bytes not in free_tensor_pool:
        free_tensor_pool[size_in_bytes] = []
    if not buffer in free_tensor_pool[size_in_bytes]:
        free_tensor_pool[size_in_bytes].append(buffer)

def download_from_gpu(buffer, shape):
    result = np.empty(dims_get_total(shape), dtype=np.float32)
    cl.enqueue_copy(queue, result, buffer)
    return result.reshape(shape)

def upload_to_gpu(array, buffer):
    cl.enqueue_copy(queue, buffer, array)

def make_gpu_buffer_from_numpy(array):
    buffer = make_array(array.shape)
    upload_to_gpu(array, buffer)
    return buffer


prg = cl.Program(ctx, """
//js
#define f32 float
#define i32 int
#define u32 unsigned int
#define ifor(N) for (i32 i = 0; i < (int)(N); ++i)
#define jfor(N) for (i32 j = 0; j < (int)(N); ++j)

__kernel void max(
    __global f32 *dst,
    __global const f32 *a,
    __global const f32 *b,
    const i32 N)
{
    const i32 gid  = get_global_id(0);
    if (gid < N) {
        dst[gid] = a[gid] > b[gid] ? a[gid] : b[gid];
    }
}
                 
__kernel void min(
    __global f32 *dst,
    __global const f32 *a,
    __global const f32 *b,
    const i32 N)
{
    const i32 gid  = get_global_id(0);
    if (gid < N) {
        dst[gid] = a[gid] < b[gid] ? a[gid] : b[gid];
    }
}

__kernel void leaky_relu(
    __global f32 *dst,
    __global const f32 *a,
    const f32 negative_slope,
    const i32 N)
{
    const i32 gid  = get_global_id(0);
    if (gid < N) {
        dst[gid] = max(a[gid], negative_slope * a[gid]);
    }
}

__kernel void leaky_relu_grad(
    __global f32 *dst,
    __global const f32 *a,
    const f32 negative_slope,
    const i32 N)
{
    const i32 gid  = get_global_id(0);
    if (gid < N) {
        dst[gid] = a[gid] < 0.0f ? negative_slope : 1.0f;
    }
}

__kernel void add(
    __global f32 *dst,
    __global const f32 *a,
    __global const f32 *b,
    const i32 N)
{
    const i32 gid  = get_global_id(0);
    if (gid < N) {
        dst[gid] = a[gid] + b[gid];
    }
}

__kernel void sub(
    __global f32 *dst,
    __global const f32 *a,
    __global const f32 *b,
    const i32 N)
{
    const i32 gid  = get_global_id(0);
    if (gid < N) {
        dst[gid] = a[gid] - b[gid];
    }
}

__kernel void mul(
    __global f32 *dst,
    __global const f32 *a,
    __global const f32 *b,
    const i32 N)
{
    const i32 gid  = get_global_id(0);
    if (gid < N) {
        dst[gid] = a[gid] * b[gid];
    }
}

__kernel void div(
    __global f32 *dst,
    __global const f32 *a,
    __global const f32 *b,
    const i32 N)
{
  const i32 gid  = get_global_id(0);
  if (gid < N) {
    dst[gid] = a[gid] / b[gid];
  }
}

__kernel void abs(
    __global f32 *dst,
    __global const f32 *a,
    const i32 N)
{
    const i32 gid  = get_global_id(0);
    if (gid < N) {
        dst[gid] = fabs(a[gid]);
    }
}
                 
__kernel void sign(
    __global f32 *dst,
    __global const f32 *a,
    const i32 N)
{
    const i32 gid  = get_global_id(0);
    if (gid < N) {
        dst[gid] = a[gid] < 0.0f ? -1.0f : (a[gid] > 0.0f ? 1.0f : 0.0f);
    }
}
                 
__kernel void sqrt_kernel(
    __global f32 *dst,
    __global const f32 *a,
    const i32 N)
{
    const i32 gid  = get_global_id(0);
    if (gid < N) {
        dst[gid] = sqrt(a[gid]);
    }
}

__kernel void addf(
    __global f32 *dst,
    __global const f32 *a,
    const f32 b,
    const i32 N)
{
    const i32 gid  = get_global_id(0);
    if (gid < N) {
        dst[gid] = a[gid] + b;
    }
}
                 
__kernel void add_first_elem(
    __global f32 *dst,     // layout [B, N]
    __global const f32 *a, // layout [B, N]
    __global const f32 *b, // layout [B, 1]
    const i32 B,
    const i32 N)
{
    const i32 gid  = get_global_id(0);
    const i32 bid  = gid / N;
    if (gid < (B * N)) {
        dst[gid] = a[gid] + b[bid];
    }
}
                 
__kernel void broadcast_add_kernel(
    __global f32 *dst,     // layout [N]
    __global const f32 *a, // layout [N]
    __global const f32 *b, // layout [1]
    const i32 N)
{
    const i32 gid  = get_global_id(0);
    if (gid < N) {
        dst[gid] = a[gid] + b[0];
    }
}

__kernel void subf(
    __global f32 *dst,
    __global const f32 *a,
    const f32 b,
    const i32 N)
{
    const i32 gid  = get_global_id(0);
    if (gid < N) {
        dst[gid] = a[gid] - b;
    }
}
                 
__kernel void mulf(
    __global f32 *dst,
    __global const f32 *a,
    const f32 b,
    const i32 N)
{
    const i32 gid  = get_global_id(0);
    if (gid < N) {
        dst[gid] = a[gid] * b;
    }
}

__kernel void divf(
    __global f32 *dst,
    __global const f32 *a,
    const f32 b,
    const i32 N)
{
    const i32 gid  = get_global_id(0);
    if (gid < N) {
        dst[gid] = a[gid] / b;
    }
}

// https://ingowald.blog/2018/06/24/float-atomics-in-opencl/
// https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
static void atomicAdd_g_f(volatile __global float *addr, float val)
{
    union {
        unsigned int u;
        float f;
    } next, expected, current;
    current.f = *addr;
    do {
        expected.f = current.f;
        next.f     = expected.f + val;
        current.u  = atomic_cmpxchg( (volatile __global unsigned int *)addr, expected.u, next.u);
    } while( current.u != expected.u );
}

// Reduce by locally sum 64 elements and then do 1 atomic add
__kernel void reduce_64(
    __global f32 *dst,
    __global const f32 *src,
    const i32 N)
{
    const i32 gid = 64 * get_global_id(0);
    f32 acc = (f32)(0.0);
    ifor(64) {
        const i32 idx = gid + i;
        if (idx < N) {
            acc += src[idx];
        }
    }

    if (gid < N) {
        atomicAdd_g_f(&dst[0], acc);
    }
}

// matmul (N, C) by (F, C)   
__kernel void matmul(
    __global f32 *dst,       // layout = [N, F]
    __global const f32 *src, // layout = [N, C]
    __global const f32 *mat, // layout = [F, C]
    const i32 N,
    const i32 C,
    const i32 F
    ) {
    const i32 gid  = get_global_id(0);
    const i32 nidx = gid / F;
    const i32 fidx = gid % F;
    
    if (gid < (N * F)) {
        f32 acc = (f32)(0.0);
        ifor(C) {
            acc += src[nidx * C + i] * mat[fidx * C + i];
        }
        dst[gid] = acc;
    }
}
            
__kernel void outer_product_accumulate(
    __global f32 *dst,       // layout = [F, C]
    __global const f32 *a,   // layout = [N, C]
    __global const f32 *b,   // layout = [N, F]
    const i32 N,
    const i32 C,
    const i32 F
    ) {
    const i32 gid  = get_global_id(0);
    const i32 fidx = gid / C;
    const i32 cidx = gid % C;
    
    if (gid < (C * F)) {
        f32 acc = (f32)(0.0);
        ifor(N) {
            acc += a[i * C + cidx] * b[i * F + fidx];
        }
        atomicAdd_g_f(&dst[gid], acc);
    }
}

__kernel void zero(
    __global f32 *dst,
    const i32 N
) {
    const i32 gid = get_global_id(0);
    if (gid < N) {
        dst[gid] = 0.0f;
    }
}

__kernel void set(
    __global f32 *dst,
    const f32 val,
    const i32 N
) {
    const i32 gid = get_global_id(0);
    if (gid < N) {
        dst[gid] = val;
    }
}

__kernel void copy(
    __global f32 *dst,
    const i32 dst_offset_elems,
    __global const f32 *src,
    const i32 src_offset_elems,
    const i32 N
) {
    const i32 gid = get_global_id(0);
    if (gid < N) {
        dst[gid + dst_offset_elems] = src[gid + src_offset_elems];
    }
}

__kernel void transpose(
    __global f32 *dst,       // layout = [C, F]
    __global const f32 *src, // layout = [F, C]
    const i32 F,
    const i32 C
) {
    const i32 gid = get_global_id(0);
    const i32 fidx = gid / C;
    const i32 cidx = gid % C;

    if (gid < (F * C)) {
        dst[cidx * F + fidx] = src[gid];
    }
}

// https://github.com/skeeto/hash-prospector
                  
static uint lowbias32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

typedef struct {
    f32 x;
    f32 y;       
} f32x2;

static f32x2 make_f32x2(f32 x, f32 y) {
    f32x2 result;
    result.x = x;
    result.y = y;
    return result;
}

static f32x2 random_f32x2(uint *seed) {
    *seed = lowbias32(*seed);
    return make_f32x2((f32)(*seed & 0xFFFF) / 0xFFFF, (f32)((*seed >> 16) & 0xFFFF) / 0xFFFF);
}

// https://gpuopen.com/learn/sampling-normal-gaussian-distribution-gpus/

#define TWO_PI (6.283185307179586476925286766559f)

static f32x2 sampleGaussBoxMuller(f32x2 u, float mean, float standardDeviation)
{
    const float a = standardDeviation * sqrt(-2.0f * log(1.0f - u.x));
    const float b = TWO_PI * u.y;

    return (f32x2){cos(b) * a + mean, sin(b) * a + mean};
}

static f32 SampleGaussian(uint *seed, const f32 mean, const f32 variance) {
    f32x2 u = random_f32x2(seed);
    return sampleGaussBoxMuller(u, mean, sqrt(variance)).x;
}

__kernel void set_gaussian(
    __global f32 *dst,
    const f32 mean,
    const f32 variance,
    const u32 rseed,
    const i32 N
) {
    const i32 gid = get_global_id(0);
    u32 seed = lowbias32(rseed + lowbias32(gid));
    if (gid < N) {
        dst[gid] = SampleGaussian(&seed, mean, variance);
    }
}

__kernel void reduce_broadcast(
    __global f32 *dst,       // [B, N]
    __global f32 const *src, // [B, M, N]
    const i32 B,
    const i32 M,
    const i32 N
) {
    const i32 gid = get_global_id(0);
    const i32 bid = gid / (N);
    const i32 nid = gid % (N);
    if (gid < (B * N)) {
        f32 acc = 0.0f;
        ifor(M) {
            acc += src[bid * M * N + i * N + nid];
        }
        dst[gid] = acc;
    }
}

;//
                 
""").build()

assert prg is not None

def roundup_to_64(x): return ((int(math.ceil(x)) + 63) // 64) * 64


_add_kernel                      = prg.add
_sub_kernel                      = prg.sub
_mul_kernel                      = prg.mul
_div_kernel                      = prg.div
_abs_kernel                      = prg.abs
_sign_kernel                     = prg.sign
_sqrt_kernel                     = prg.sqrt_kernel
_addf_kernel                     = prg.addf
_add_first_elem_kernel           = prg.add_first_elem
_subf_kernel                     = prg.subf
_mulf_kernel                     = prg.mulf
_divf_kernel                     = prg.divf
_reduce_64_kernel                = prg.reduce_64
_matmul_kernel                   = prg.matmul
_outer_product_accumulate_kernel = prg.outer_product_accumulate
_transpose_kernel                = prg.transpose
_zero_kernel                     = prg.zero
_set_kernel                      = prg.set
_set_gaussian_kernel                      = prg.set_gaussian
_min_kernel                      = prg.min
_max_kernel                      = prg.max
_leaky_relu_grad_kernel          = prg.leaky_relu_grad
_leaky_relu_kernel               = prg.leaky_relu
_copy_kernel                     = prg.copy
_reduce_broadcast_kernel         = prg.reduce_broadcast
_broadcast_add_kernel               = prg.broadcast_add_kernel


def add_kernel(dst, a, b, N): _add_kernel(queue, [roundup_to_64(N)], None, dst, a, b, np.array([N], dtype=np.int32))
def sub_kernel(dst, a, b, N): _sub_kernel(queue, [roundup_to_64(N)], None, dst, a, b, np.array([N], dtype=np.int32))
def mul_kernel(dst, a, b, N): _mul_kernel(queue, [roundup_to_64(N)], None, dst, a, b, np.array([N], dtype=np.int32))
def div_kernel(dst, a, b, N): _div_kernel(queue, [roundup_to_64(N)], None, dst, a, b, np.array([N], dtype=np.int32))
def abs_kernel(dst, a, N): _abs_kernel(queue, [roundup_to_64(N)], None, dst, a, np.array([N], dtype=np.int32))
def sign_kernel(dst, a, N): _sign_kernel(queue, [roundup_to_64(N)], None, dst, a, np.array([N], dtype=np.int32))
def sqrt_kernel(dst, a, N): _sqrt_kernel(queue, [roundup_to_64(N)], None, dst, a, np.array([N], dtype=np.int32))
def addf_kernel(dst, a, b, N): _addf_kernel(queue, [roundup_to_64(N)], None, dst, a, np.array([b], dtype=np.float32), np.array([N], dtype=np.int32))
def add_first_elem_kernel(dst, a, b, B, N): _add_first_elem_kernel(queue, [roundup_to_64(B * N)], None, dst, a, b, np.array([B], dtype=np.int32), np.array([N], dtype=np.int32))
def broadcast_add_kernel(dst, a, b, N): _broadcast_add_kernel(queue, [roundup_to_64(N)], None, dst, a, b, np.array([N], dtype=np.int32))
def subf_kernel(dst, a, b, N): _subf_kernel(queue, [roundup_to_64(N)], None, dst, a, np.array([b], dtype=np.float32), np.array([N], dtype=np.int32))
def mulf_kernel(dst, a, b, N): _mulf_kernel(queue, [roundup_to_64(N)], None, dst, a, np.array([b], dtype=np.float32), np.array([N], dtype=np.int32))
def divf_kernel(dst, a, b, N): _divf_kernel(queue, [roundup_to_64(N)], None, dst, a, np.array([b], dtype=np.float32), np.array([N], dtype=np.int32))
def reduce_kernel(dst, src, N): _reduce_64_kernel(queue, [roundup_to_64(N / 64)], None, dst, src, np.array([N], dtype=np.int32))
def matmul_kernel(dst, a, b, N, C, F): _matmul_kernel(queue, [roundup_to_64(N * F)], None, dst, a, b, np.array([N]), np.array([C], dtype=np.int32), np.array([F], dtype=np.int32))
def outer_product_accumulate_kernel(dst, a, b, N, C, F): _outer_product_accumulate_kernel(queue, [roundup_to_64(F * C)], None, dst, a, b, np.array([N]), np.array([C]), np.array([F]))
def transpose_kernel(dst, src, F, C): _transpose_kernel(queue, [roundup_to_64(F * C)], None, dst, src, np.array([F], dtype=np.int32), np.array([C], dtype=np.int32))
def set_kernel(dst, val, N): _set_kernel(queue, [roundup_to_64(N)], None, dst, np.array([val], dtype=np.float32), np.array([N], dtype=np.int32))
def zero_kernel(dst, N): _zero_kernel(queue, [roundup_to_64(N)], None, dst, np.array([N], dtype=np.int32))
def set_gaussian_kernel(dst, mean, variance, seed,  N): _set_gaussian_kernel(queue, [roundup_to_64(N)], None, dst, np.array([mean], dtype=np.float32), np.array([variance], dtype=np.float32), np.array([seed], dtype=np.int32), np.array([N], dtype=np.int32))
def min_kernel(dst, a, b, N): _min_kernel(queue, [roundup_to_64(N)], None, dst, a, b, np.array([N], dtype=np.int32))
def max_kernel(dst, a, b, N): _max_kernel(queue, [roundup_to_64(N)], None, dst, a, b, np.array([N], dtype=np.int32))
def leaky_relu_grad_kernel(dst, a, negative_slope, N): _leaky_relu_grad_kernel(queue, [roundup_to_64(N)], None, dst, a, np.array([negative_slope], dtype=np.float32), np.array([N], dtype=np.int32))
def leaky_relu_kernel(dst, a, negative_slope, N): _leaky_relu_kernel(queue, [roundup_to_64(N)], None, dst, a, np.array([negative_slope], dtype=np.float32), np.array([N], dtype=np.int32))
def copy_kernel(dst, dst_offset_elems, src, src_offset_elems, N): _copy_kernel(queue, [roundup_to_64(N)], None, dst, np.array([dst_offset_elems], dtype=np.int32), src, np.array([src_offset_elems], dtype=np.int32), np.array([N], dtype=np.int32))
def reduce_broadcast(dst, src, B, M, N): _reduce_broadcast_kernel(queue, [roundup_to_64(B * N)], None, dst, src, np.array([B], dtype=np.int32), np.array([M], dtype=np.int32), np.array([N], dtype=np.int32))

"""
New basic building block class for compute.
Basically a flat array with a rule for accessing elements.
We're using a basic rule of linear strides.
"""
class Tensor:
    def __init__(self, shape, data=None):
        self.shape   = list(shape)
        if data is not None:
            if isinstance(data, list):
                data = np.array(data, dtype=np.float32)
            assert dims_get_total(shape) == dims_get_total(list(data.shape)), f"Data shape {data.shape} does not match tensor shape {shape}"
            assert isinstance(data, np.ndarray), f"Data must be a numpy array, got {type(data)}"
            data = data.astype(np.float32)
            self.buffer = make_gpu_buffer_from_numpy(data)
        else:
            self.buffer = make_array(shape)
            zero_kernel(self.buffer, dims_get_total(shape))

        self.strides = []
        self._compute_strides()

    def download(self): return download_from_gpu(self.buffer, self.shape)

    def _compute_strides(self):
        total = 1
        for d in reversed(self.shape):
            self.strides.insert(0, total)
            total *= d

    def _get_flat_index(self, indices):
        if len(indices) != len(self.shape):
            raise ValueError("Incorrect number of indices")
        index = 0
        for i, idx in enumerate(indices):
            if not (0 <= idx < self.shape[i]):
                raise IndexError("Index out of bounds")
            index += idx * self.strides[i]
        return index

    def __repr__(self):
        return f"Tensor(shape={self.shape})"
    
    def __add__(self, other):
        result = Tensor(self.shape)
        if isinstance(other, (int, float)):
            addf_kernel(result.buffer, self.buffer, other, dims_get_total(self.shape))
        else:
            assert self.shape == other.shape, f"Shapes {self.shape} and {other.shape} must match"
            add_kernel(result.buffer, self.buffer, other.buffer, dims_get_total(self.shape))
        return result
    
    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            addf_kernel(self.buffer, self.buffer, other, dims_get_total(self.shape))
        else:
            assert self.shape == other.shape, f"Shapes {self.shape} and {other.shape} must match"
            add_kernel(self.buffer, self.buffer, other.buffer, dims_get_total(self.shape))
        return self

    def __mul__(self, other):
        result = Tensor(self.shape)
        if isinstance(other, (int, float)):
            mulf_kernel(result.buffer, self.buffer, other, dims_get_total(self.shape))
        else:
            assert self.shape == other.shape, f"Shapes {self.shape} and {other.shape} must match"
            mul_kernel(result.buffer, self.buffer, other.buffer, dims_get_total(self.shape))
        return result

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            addf_kernel(self.buffer, self.buffer, other, dims_get_total(self.shape))
        else:
            assert self.shape == other.shape, f"Shapes {self.shape} and {other.shape} must match"
            add_kernel(self.buffer, self.buffer, other.buffer, dims_get_total(self.shape))
        return self

    def __sub__(self, other):
        result = Tensor(self.shape)
        if isinstance(other, (int, float)):
            subf_kernel(result.buffer, self.buffer, other, dims_get_total(self.shape))
        else:
            assert self.shape == other.shape, f"Shapes {self.shape} and {other.shape} must match"
            sub_kernel(result.buffer, self.buffer, other.buffer, dims_get_total(self.shape))
        return result
    
    def __isub__(self, other):
        if isinstance(other, (int, float)):
            subf_kernel(self.buffer, self.buffer, other, dims_get_total(self.shape))
        else:
            assert self.shape == other.shape, f"Shapes {self.shape} and {other.shape} must match"
            sub_kernel(self.buffer, self.buffer, other.buffer, dims_get_total(self.shape))
        return self

    def __truediv__(self, other):
        result = Tensor(self.shape)
        if isinstance(other, (int, float)):
            divf_kernel(result.buffer, self.buffer, other, dims_get_total(self.shape))
        else:
            assert self.shape == other.shape, f"Shapes {self.shape} and {other.shape} must match"
            div_kernel(result.buffer, self.buffer, other.buffer, dims_get_total(self.shape))
        return result
    
    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            divf_kernel(self.buffer, self.buffer, other, dims_get_total(self.shape))
        else:
            assert self.shape == other.shape, f"Shapes {self.shape} and {other.shape} must match"
            div_kernel(self.buffer, self.buffer, other.buffer, dims_get_total(self.shape))
        return self

    def abs(self):
        result = Tensor(self.shape)
        abs_kernel(result.buffer, self.buffer, dims_get_total(self.shape))
        return result
    
    def sqrt(self):
        result = Tensor(self.shape)
        sqrt_kernel(result.buffer, self.buffer, dims_get_total(self.shape))
        return result

if 1: # tests
    #########
    # TESTS #
    #########
    
    a = make_gpu_buffer_from_numpy(np.array([1, 1, 1, 1], dtype=np.float32))
    b = make_gpu_buffer_from_numpy(np.array([1, 1, 1, 2], dtype=np.float32))
    c = make_array((4,))

    add_kernel(c, a, b, a.size)

    d = make_array((4,))

    reduce_kernel(d, c, a.size)

    b = download_from_gpu(d, (1,))

    # print(b)
    assert b[0] == 9.0, f"Reduction result is wrong! {b[0]} vs 9.0"

    N           = 4
    C           = 4
    F           = 8
    np_a        = np.random.rand(N, C).astype(np.float32)
    np_mat      = np.random.rand(F, C).astype(np.float32)
    np_result   = np.einsum("nc, fc -> nf", np_a, np_mat)

    a   = make_gpu_buffer_from_numpy(np_a)
    mat = make_gpu_buffer_from_numpy(np_mat)
    c   = make_array((N, F))

    matmul_kernel(c, a, mat, N, C, F)

    b = download_from_gpu(c, (N, F))

    assert (np.abs(b - np_result) < 1e-4).all(), f"Results do not match! {b} vs {np_result}"
    free_all_buffers()

    # outer product accumulation

    N           = 4
    C           = 4
    F           = 8
    np_a        = np.random.rand(N, C).astype(np.float32)
    np_b        = np.random.rand(N, F).astype(np.float32)
    np_result   = np.einsum("nc, nf -> fc", np_a, np_b)

    a   = make_gpu_buffer_from_numpy(np_a)
    b   = make_gpu_buffer_from_numpy(np_b)
    c   = make_array((F, C))
    zero_kernel(c, F * C)
    outer_product_accumulate_kernel(c, a, b, N, C, F)

    b = download_from_gpu(c, (F, C))

    assert (np.abs(b - np_result) < 1e-4).all(), f"Results do not match!\n{b}\n vs\n{np_result}\n"
    free_all_buffers()

    # transpose
    np_mat = np.random.rand(F, C).astype(np.float32)
    np_result = np_mat.T
    a = make_gpu_buffer_from_numpy(np_mat)
    c = make_array((C, F))
    zero_kernel(c, F * C)
    transpose_kernel(c, a, F, C)

    b = download_from_gpu(c, (C, F))

    assert (np.abs(b - np_result) < 1e-4).all(), f"Results do not match! {b} vs {np_result}"
    free_all_buffers()

    B = 32

    np_a = np.random.rand(B, 16)
    np_b = np.random.rand(B, 16)
    a = Tensor(shape=[B, 16], data=np_a)
    b = Tensor(shape=[B, 16], data=np_b)

    assert (np.abs(b.download() - np_b) < 1e-4).all(), f"Tensor download failed! {b.download()} vs {np_b}"
    assert (np.abs(a.download() - np_a) < 1e-4).all(), f"Tensor download failed! {a.download()} vs {np_a}"
    
    np_a = np.random.rand(B, 16)
    np_b = np.random.rand(B, 16)
    a = Tensor(shape=[B, 16], data=np_a)
    b = Tensor(shape=[B, 16], data=np_b)
    ref = a.download() + b.download()
    c   = a + b
    assert (np.abs(c.download() - ref) < 1e-4).all(), f"Tensor addition failed! {c.download()} vs {ref}"
    free_all_buffers()
    
    np_a = np.random.rand(B, 16)
    np_b = np.random.rand(B, 16)
    a = Tensor(shape=[B, 16], data=np_a)
    b = Tensor(shape=[B, 16], data=np_b)
    ref = a.download() - b.download()
    c   = a - b
    assert (np.abs(c.download() - ref) < 1e-4).all(), f"Tensor subtraction failed! {c.download()} vs {ref}"
    free_all_buffers()

    np_a = np.random.rand(B, 16)
    np_b = np.random.rand(B, 16)
    a = Tensor(shape=[B, 16], data=np_a)
    b = Tensor(shape=[B, 16], data=np_b)
    ref = a.download() * b.download()
    c   = a * b
    assert (np.abs(c.download() - ref) < 1e-4).all(), f"Tensor multiplication failed! {c.download()} vs {ref}"
    free_all_buffers()

    np_a = np.random.rand(B, 16)
    np_b = np.random.rand(B, 16) * 16.0
    a = Tensor(shape=[B, 16], data=np_a)
    b = Tensor(shape=[B, 16], data=np_b)
    ref = a.download() / b.download()
    c   = a / b
    assert (np.abs(c.download() - ref) < 1e-3).all(), f"Tensor division failed! {c.download()} vs {ref}"
    free_all_buffers()

    np_a = np.random.rand(B, 16)
    np_b = np.random.rand(B, 16)
    a = Tensor(shape=[B, 16], data=np_a)
    b = Tensor(shape=[B, 16], data=np_b)
    ref = a.download() + 0.777
    c   = a + 0.777
    assert (np.abs(c.download() - ref) < 1e-4).all(), f"Tensor addition failed! {c.download()} vs {ref}"
    free_all_buffers()

    np_a = np.random.rand(B, 16)
    np_b = np.random.rand(B, 16)
    a = Tensor(shape=[B, 16], data=np_a)
    b = Tensor(shape=[B, 16], data=np_b)
    ref = a.download() - 0.777
    c   = a - 0.777
    assert (np.abs(c.download() - ref) < 1e-4).all(), f"Tensor subtraction failed! {c.download()} vs {ref}"
    free_all_buffers()

    np_a = np.random.rand(B, 16)
    np_b = np.random.rand(B, 16)
    a = Tensor(shape=[B, 16], data=np_a)
    b = Tensor(shape=[B, 16], data=np_b)
    ref = a.download() * 0.777
    c   = a * 0.777
    assert (np.abs(c.download() - ref) < 1e-4).all(), f"Tensor multiplication failed! {c.download()} vs {ref}"
    free_all_buffers()

    np_a = np.random.rand(B, 16)
    np_b = np.random.rand(B, 16)
    a = Tensor(shape=[B, 16], data=np_a)
    b = Tensor(shape=[B, 16], data=np_b)
    ref = a.download() / 0.777
    c   = a / 0.777
    assert (np.abs(c.download() - ref) < 1e-4).all(), f"Tensor division failed! {c.download()} vs {ref}"
    free_all_buffers()

    np_a = np.random.rand(B, 16)
    ref = np.abs(np_a)
    a = Tensor(shape=[B, 16], data=np_a)
    c = a.abs()
    assert (np.abs(c.download() - ref) < 1e-4).all(), f"Tensor absolute failed! {c.download()} vs {ref}"

    E = 7
    np_a = np.random.rand(B, 16)
    ref = np.ndarray((B, E * 16))

    for i in range(E):
        for j in range(16):
            ref[:, i * 16 + j] = np_a[:, j]

    a = Tensor(shape=[B, 16], data=np_a)
    c = Tensor(shape=[B, 16 * E])

    for b in range(B):
        for i in range(16): # Clone the tensor N times
            copy_kernel(c.buffer, b * 16 * E + i * 16, a.buffer, b * 16, 16)

    assert (np.abs(c.download() - ref) < 1e-4).all(), f"Tensor broadcast failed! {c.download()} vs {ref}"

    ref_reduce = np.zeros((B, 16))
    for b in range(B):
        for i in range(E):
            for j in range(16):
                ref_reduce[b, j] += ref[b, i * 16 + j]

    b = Tensor(shape=[B, 16])
    reduce_broadcast(b.buffer, c.buffer, B, E, 16)

    assert (np.abs(b.download() - ref_reduce) < 1e-4).all(), f"Tensor reduction failed! {b.download()} vs {ref_reduce}"
    free_all_buffers()

    np_a = np.random.normal(0, 1, (B, 16))
    np_b = np.random.normal(0, 1, (B, 1))
    ref = np.zeros((B, 16))
    for b in range(B):
        ref[b] = np_a[b] + np_b[b]

    a = Tensor(shape=[B, 16], data=np_a)
    b = Tensor(shape=[B, 1], data=np_b)
    c = Tensor(shape=[B, 16])
    add_first_elem_kernel(c.buffer, a.buffer, b.buffer, B, 16)

    assert (np.abs(c.download() - ref) < 1e-4).all(), f"Tensor addition with first elem failed! {c.download()} vs {ref}"
    free_all_buffers()

    np_a = np.random.normal(0, 1, (B, 16))
    ref = np.sqrt(np.abs(np_a))

    a = Tensor(shape=[B, 16], data=np_a)
    c = Tensor(shape=[B, 16])
    abs_kernel(c.buffer, a.buffer, B * 16)
    sqrt_kernel(c.buffer, c.buffer, B * 16)

    assert (np.abs(c.download() - ref) < 1e-4).all(), f"Tensor sqrt failed! {c.download()} vs {ref}"
    free_all_buffers()

    np_a = np.random.normal(0, 1, (B, 16))
    ref = np.maximum(np_a, 0.1 * np_a)

    a = Tensor(shape=[B, 16], data=np_a)
    c = Tensor(shape=[B, 16])
    leaky_relu_kernel(c.buffer, a.buffer, 0.1, B * 16)

    assert (np.abs(c.download() - ref) < 1e-4).all(), f"Tensor leaky_relu failed! {c.download()} vs {ref}"
    free_all_buffers()

    np_a = np.random.normal(0, 1, (B, 16))
    ref = np.where(np_a > 0, 1.0, 0.1)

    a = Tensor(shape=[B, 16], data=np_a)
    c = Tensor(shape=[B, 16])
    leaky_relu_grad_kernel(c.buffer, a.buffer, 0.1, B * 16)

    assert (np.abs(c.download() - ref) < 1e-4).all(), f"Tensor leaky_relu failed! {c.download()} vs {ref}"
    free_all_buffers()

    np_a = np.random.normal(0, 1, (B, 16))
    ref  = np.where(np_a > 0, 1.0, 0.1)
    a = Tensor(shape=[B, 16], data=np_a)
    c = Tensor(shape=[B, 16])
    set_kernel(c.buffer, 777.0, B * 16)
    ref = np.full((B, 16), 777.0, dtype=np.float32)

    assert (np.abs(c.download() - ref) < 1e-4).all(), f"Tensor set failed! {c.download()} vs {ref}"
    free_all_buffers()

    np_a = np.random.normal(0, 1, (B, 16))
    np_b = np.random.normal(0, 1, (B, 16))
    ref  = np_a / (np_b + 1.0e-6)
    a = Tensor(shape=[B, 16], data=np_a)
    b = Tensor(shape=[B, 16], data=np_b)
    c = a / (b + 1.0e-6)

    assert (np.abs(c.download() - ref) < 1e-3).all(), f"Tensor failed! {c.download()} vs {ref}"
    free_all_buffers()

    np_a = np.random.normal(0, 1, (B, 16))
    np_b = np.random.normal(0, 1, (B, 16))
    ref  = np_a / (np.sqrt(np.abs(np_b)) + 1.0e-6)
    a = Tensor(shape=[B, 16], data=np_a)
    b = Tensor(shape=[B, 16], data=np_b)
    c = a / (b.abs().sqrt() + 1.0e-6)

    assert (np.abs(c.download() - ref) < 1e-4).all(), f"Tensor failed! {c.download()} vs {ref}"
    free_all_buffers()

    np_a = np.random.normal(0, 1, (16, 16))
    np_b = np.random.normal(0, 1, (16, 16))
    np_c = np.random.normal(0, 1, (16, 16))
    ref  = np_a / (np.sqrt(np.abs(np_b)) + 1.0e-6) - np_c * 0.001
    a = Tensor(shape=[16, 16], data=np_a)
    b = Tensor(shape=[16, 16], data=np_b)
    c = Tensor(shape=[16, 16], data=np_c)
    c = a / (b.abs().sqrt() + 1.0e-6) - c * 0.001

    assert (np.abs(c.download() - ref) < 1e-4).all(), f"Tensor failed! {c.download()} vs {ref}"
    free_all_buffers()
    
    print(f"{TERMINAL_COLOR_GREEN}All tests passed!{TERMINAL_COLOR_RESET}")
    
    # exit()

# Compute graph basic building block
class AutoGradNode:
    def __init__(self, shape):
        # scalar valued gradient accumulator for the final dL/dp
        self.shape = shape
        self.grad  = Tensor(shape)
        # dependencies for causation sort
        self.dependencies = []
        self.materialized = None

    def zero_grad(self): zero_kernel(self.grad.buffer, dims_get_total(self.grad.shape))

    # Overload operators to build the computation graph
    def __add__(self, other): return Add(self, other)
    def __mul__(self, other): return Mul(self, other)
    def __sub__(self, other): return Sub(self, other)

    # Get a topologically sorted list of dependencies
    # starts from the leaf nodes and terminates at the root
    def get_topo_sorted_list_of_deps(self):
        visited = set()
        topo_order = []

        def dfs(node): # depth-first search
            if node in visited:
                return
            visited.add(node)
            for dep in node.dependencies:
                dfs(dep)
            topo_order.append(node)

        dfs(self)

        return topo_order

    def get_pretty_name(self): return self.__class__.__name__

    # Pretty print the computation graph in DOT format
    def pretty_print_dot_graph(self):
        topo_order = self.get_topo_sorted_list_of_deps()
        _str = ""
        _str += "digraph G {\n"
        for node in topo_order:
            _str += f"    {id(node)} [label=\"{node.get_pretty_name()}\"];\n"
            for dep in node.dependencies:
                _str += f"    {id(node)} -> {id(dep)};\n"
        _str += "}"
        return _str
    
    def backward(self):
        topo_order = self.get_topo_sorted_list_of_deps()

        for node in topo_order:
            node.zero_grad() # we don't want to accumulate gradients

        set_kernel(self.grad.buffer, 1.0, dims_get_total(self.grad.shape)) # seed the gradient at the output node dL/dL=1

        # Reverse the topological order for backpropagation to start from the output
        for node in reversed(topo_order):
            # from the tip of the  graph down to leaf learnable parameters
            # Distribute gradients
            node._backward()

    # The job of this method is to propagate gradients backward through the network
    def _backward(self):
        assert False, "Not implemented in base class"

    # Materialize the numerical value at the node
    # i.e. Evaluate the computation graph
    def _materialize(self):
        assert False, "Not implemented in base class"

    # Cache
    def materialize(self):
        if self.materialized is None:
            self.materialized = self._materialize()
            assert self.materialized is not None, f"Materialization failed for {self.get_pretty_name()}"
            # print(f"Materialized {self.get_pretty_name()}: {self.materialized.shape}")
        return self.materialized

# Any value that is not learnable
class Variable(AutoGradNode):
    def __init__(self, values, name=None):
        assert isinstance(values, Tensor), f"Values must be a Tensor, got {type(values)}"
        super().__init__(shape=values.shape)
        self.values = values
        self.name   = name

    def get_pretty_name(self):
        if self.name:
            return f"Variable({self.name})"
        else:
            return f"Constant()"

    def _materialize(self): return self.values

    def _backward(self):
        pass

Constant = Variable

# Learnable parameter with initial random value 0..1
class LearnableParameter(AutoGradNode):
    def __init__(self, shape):
        super().__init__(shape=shape)
        self.values = Tensor(shape)
        set_gaussian_kernel(self.values.buffer, 0.0, 1.0 / shape[-1], random.randint(0, 2**10 - 1), dims_get_total(self.values.shape))

    def get_pretty_name(self):
        return f"LearnableParameter(shape={self.values.shape})"

    def _materialize(self): return self.values

    def _backward(self):
        pass

class Matrix(AutoGradNode):
    def __init__(self, in_channels, out_features):
        super().__init__(shape=[out_features, in_channels])
        self.values = Tensor([out_features, in_channels])
        set_gaussian_kernel(self.values.buffer, 0.0, 1.0 / in_channels, random.randint(0, 2**10 - 1), dims_get_total(self.values.shape))

    def get_pretty_name(self):
        return f"Matrix({self.values.shape})"

    def _materialize(self):
        return self.values

    def _backward(self):
        pass

    def transposed(self):
        in_features  = self.values.shape[1]
        out_features = self.values.shape[0]
        transposed   = Matrix(in_channels=out_features, out_features=in_features)
        transpose_kernel(transposed.values.buffer, self.values.buffer, out_features, in_features)
        return transposed

def tensor_matrix_multiply(tensor, matrix):
    in_features           = tensor.shape[-1]
    out_features          = matrix.shape[0]
    assert in_features == matrix.shape[1], f"Incompatible matrix dimensions {tensor.shape} and {matrix.shape}"
    result = Tensor(shape=tensor.shape[:-1] + [out_features])
    matmul_kernel(result.buffer, tensor.buffer, matrix.buffer, dims_get_total(tensor.shape) // in_features, in_features, out_features)
    return result

def tensor_outer_product(tensor_a, tensor_b):
    in_features             = tensor_a.shape[-1]
    out_features            = tensor_b.shape[-1]
    result_shape            = [out_features, in_features] # tensor_a.shape[:-1] + [out_features, in_features]
    total_number_of_elems_a = dims_get_total(tensor_a.shape) # we don't really care about the actual shape, for this we know that the tensor is an array of input features
    total_number_of_elems_b = dims_get_total(tensor_b.shape) # we don't really care about the actual shape, for this we know that the tensor is an array of output features
    assert total_number_of_elems_a // in_features == total_number_of_elems_b // out_features, "Incompatible tensor dimensions for outer product"
    result                  = Tensor(shape=result_shape)
    outer_product_accumulate_kernel(result.buffer, tensor_a.buffer, tensor_b.buffer, total_number_of_elems_a // in_features, in_features, out_features)
    # outer_product_accumulate_kernel(result.buffer, tensor_b.buffer, tensor_a.buffer, total_number_of_elems_a // in_features, out_features, in_features)
    return result

def cpu_tensor_matrix_multiply(tensor, matrix):
    in_features           = tensor.shape[-1]
    out_features          = matrix.shape[0]
    assert in_features == matrix.shape[1], f"Incompatible matrix dimensions {tensor.shape} and {matrix.shape}"
    total_number_of_elems = tensor.size # we don't really care about the actual shape, for this we know that the tensor is an array of input features
    np_tensor             = tensor.reshape(total_number_of_elems // in_features, in_features)
    np_matrix             = matrix.reshape(out_features, in_features)
    np_result             = np.einsum('si,oi->so', np_tensor, np_matrix)
    return np_result

def cpu_tensor_outer_product(tensor_a, tensor_b):
    in_features   = tensor_a.shape[-1]
    out_features  = tensor_b.shape[-1]
    result_shape  = [out_features, in_features] # tensor_a.shape[:-1] + [out_features, in_features]
    total_number_of_elems_a = tensor_a.size # we don't really care about the actual shape, for this we know that the tensor is an array of input features
    total_number_of_elems_b = tensor_b.size # we don't really care about the actual shape, for this we know that the tensor is an array of output features
    assert total_number_of_elems_a // in_features == total_number_of_elems_b // out_features, "Incompatible tensor dimensions for outer product"

    np_tensor_a = tensor_a.reshape(total_number_of_elems_a // in_features, in_features)
    np_tensor_b = tensor_b.reshape(total_number_of_elems_b // out_features, out_features)
    np_result   = np.einsum('si,sj->ji', np_tensor_a, np_tensor_b)
    return np_result


class VectorMatrixMultiply(AutoGradNode):
    def __init__(self, tensor, matrix):
        assert tensor.shape[-1] == matrix.shape[1], f"Incompatible matrix dimensions: {tensor.shape} vs {matrix.shape}"
        super().__init__(shape=tensor.shape[:-1] + [matrix.shape[0]])
        self.tensor       = tensor
        self.matrix       = matrix
        self.dependencies = [tensor, matrix]

    def _materialize(self):
        tensor                = self.tensor.materialize()
        matrix                = self.matrix.materialize()

        if DEBUG_CPU_EMULATION: # cpu emulation
            tensor = tensor.download()
            matrix = matrix.download()
            result = cpu_tensor_matrix_multiply(tensor, matrix)
            return Tensor(shape=result.shape, data=result)
        else:
            return tensor_matrix_multiply(tensor, matrix)

    def _backward(self):
        if DEBUG_CPU_EMULATION: # cpu emulation
            grad = self.grad.download()
            mT   = self.matrix.materialize().download().T
            tensor = self.tensor.materialize().download()

            tensor_grad = cpu_tensor_matrix_multiply(grad, mT)
            weight_grad = cpu_tensor_outer_product(tensor, grad)

            self.tensor.grad = self.tensor.grad + Tensor(shape=tensor_grad.shape, data=tensor_grad)
            self.matrix.grad = self.matrix.grad + Tensor(shape=weight_grad.shape, data=weight_grad)
            # self.matrix.grad = self.matrix.grad + tensor_outer_product(self.tensor.materialize(), self.grad)

        else:

            self.tensor.grad = self.tensor.grad + tensor_matrix_multiply(self.grad, self.matrix.transposed().materialize())
            self.matrix.grad = self.matrix.grad + tensor_outer_product(self.tensor.materialize(), self.grad)

class LeakyRelu(AutoGradNode):
    def __init__(self, a, negative_slope=0.01):
        super().__init__(shape=a.shape)
        self.a              = a
        self.negative_slope = negative_slope
        self.dependencies   = [a]

    def _materialize(self):
        x           = self.a.materialize()
        result      = Tensor(x.shape)
        leaky_relu_kernel(result.buffer, x.buffer, self.negative_slope, dims_get_total(x.shape))
        return result

    def _backward(self):
        am            = self.a.materialize()
        slope         = Tensor(self.a.shape)
        leaky_relu_grad_kernel(slope.buffer, am.buffer, self.negative_slope, dims_get_total(am.shape))
        self.a.grad   = self.a.grad + self.grad * slope

class Reduce(AutoGradNode):
    def __init__(self, a, op='+'):
        super().__init__(shape=[1,])
        self.a            = a
        self.dependencies = [a]
        self.op           = op
        assert op in ['+'], "Only sum reduction is supported"

    def _materialize(self): 
        result = Tensor([1,])
        reduce_kernel(result.buffer, self.a.materialize().buffer, dims_get_total(self.a.shape))
        return result

    def _backward(self):
        broadcast_add_kernel(self.a.grad.buffer, self.a.grad.buffer, self.grad.buffer, dims_get_total(self.a.shape))

class Square(AutoGradNode):
    def __init__(self, a):
        super().__init__(shape=a.shape)
        self.a            = a
        self.dependencies = [a]

    def _materialize(self):
        x = self.a.materialize()
        return x * x

    def _backward(self):
        self.a.grad = self.a.grad + self.grad * 2.0 * self.a.materialize()

class Abs(AutoGradNode):
    def __init__(self, a):
        super().__init__(shape=a.shape)
        self.a            = a
        self.dependencies = [a]

    def _materialize(self):
        x = self.a.materialize()
        result = Tensor(x.shape)
        abs_kernel(result.buffer, x.buffer, dims_get_total(x.shape))
        return result

    def _backward(self):
        sign_tensor = Tensor(self.a.shape)
        sign_kernel(sign_tensor.buffer, self.a.materialize().buffer, dims_get_total(self.a.shape))
        self.a.grad = self.a.grad + self.grad * sign_tensor

class Sub(AutoGradNode):
    def __init__(self, a, b):
        assert a.shape == b.shape, f"Incompatible tensor dimensions: {a.shape} vs {b.shape}"
        super().__init__(shape=a.shape)
        self.a            = a
        self.b            = b
        self.dependencies = [a, b]

    def _materialize(self):
        return self.a.materialize() - self.b.materialize()

    def _backward(self):
        self.a.grad = self.a.grad + self.grad
        self.b.grad = self.b.grad - self.grad

class Add(AutoGradNode):
    def __init__(self, a, b):
        assert a.shape == b.shape, f"Incompatible tensor dimensions: {a.shape} vs {b.shape}"
        super().__init__(shape=a.shape)
        self.a = a
        self.b = b
        self.dependencies = [a, b]

    def _materialize(self):
        return self.a.materialize() + self.b.materialize()

    def _backward(self):
        self.a.grad = self.a.grad + self.grad
        self.b.grad = self.b.grad + self.grad

class Mul(AutoGradNode):
    def __init__(self, a, b):
        assert a.shape == b.shape, f"Incompatible tensor dimensions: {a.shape} vs {b.shape}"
        super().__init__(shape=a.shape)
        self.a = a
        self.b = b
        self.dependencies = [a, b]

    def _materialize(self):
        return self.a.materialize() * self.b.materialize()

    def _backward(self):
        self.a.grad = self.a.grad + self.grad * self.b.materialize()
        self.b.grad = self.b.grad + self.grad * self.a.materialize()

class Broadcast(AutoGradNode):
    def __init__(self, a, dim, size):
        assert dim == 0, f"Only broadcasting in dimension 0 is supported, got {dim}"
        input_shape = a.shape[:]
        assert input_shape[dim] == 1, f"Input tensor must have size 1 in dimension {dim}"
        input_shape[dim] = size

        super().__init__(shape=input_shape)
        self.a            = a
        self.dim          = dim
        self.size         = size
        self.dependencies = [a]

    def _materialize(self):
        # Only support [1, N] -> [B, N]
        B, N = self.a.shape
        assert B == 1, f"Input tensor must have batch size 1, got {B}"
        M = self.size
        ma          = self.a.materialize()
        result      = Tensor(self.shape)

        if DEBUG_CPU_EMULATION:
            manb = ma.download()
            ranb = result.download()
            for i in range(self.size): # Clone the tensor N times
                ctypes.memmove(ranb.ctypes.data + i * manb.nbytes, manb.ctypes.data, manb.nbytes)
            result = Tensor(self.shape, data=ranb)
            return result
        else:
            # l           = dims_get_total(self.a.shape)

            for i in range(M): # Clone the tensor N times
                copy_kernel(result.buffer, i * N, ma.buffer, 0, N)

            # for i in range(self.size): # Clone the tensor N times
            #     copy_kernel(result.buffer, i * l, ma.buffer, 0, l)
            return result

    def _backward(self):
        # Reduce the gradient
        # Only support [B, N]
        B, N   = self.a.shape
        assert B == 1, f"Input tensor must have batch size 1, got {B}"
        M      = self.size
        result = Tensor(shape=[1, N])

        if DEBUG_CPU_EMULATION:
            grad = self.grad.download()
            acc  = np.zeros((1, N), dtype=np.float32)
            for i in range(M):
                acc[0] += grad[i]
            self.a.grad = self.a.grad + Tensor(shape=[1, N], data=acc)
            return
        else:
            reduce_broadcast(result.buffer, self.grad.buffer, 1, M, N)
            self.a.grad = self.a.grad + result


class AdamW:
    """
        Simple AdamW without variance bias correction
    """
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.moments_1 = []
        self.moments_2 = []
        for p in parameters:
            self.moments_1.append(Tensor(p.grad.shape))
            self.moments_2.append(Tensor(p.grad.shape))

        
    def step(self):
        for i, p in enumerate(self.parameters):
            self.moments_1[i] = self.moments_1[i] * self.betas[0] + p.grad * (1 - self.betas[0])
            self.moments_2[i] = self.moments_2[i] * self.betas[1] + (p.grad * p.grad) * (1 - self.betas[1])
            variance          = self.moments_2[i] - self.moments_1[i] * self.moments_1[i]
            abs_kernel(variance.buffer, variance.buffer, dims_get_total(variance.shape))
            sqrt_kernel(variance.buffer, variance.buffer, dims_get_total(variance.shape))
            p.values    -= self.moments_1[i] / (variance + 1e-8) * self.lr
            p.values    -= p.values * self.weight_decay * self.lr

    def get_list_of_live_buffers(self):
        return [p.values.buffer for p in self.parameters] + [p.grad.buffer for p in self.parameters] + [p.buffer for p in self.moments_1] + [p.buffer for p in self.moments_2]


if 0: # validation


    a = LearnableParameter(shape=[1, 3])
    b = LearnableParameter(shape=[1, 3])
    m0 = Matrix(in_channels=3, out_features=3)

    adamw = AdamW(parameters=[a, b, m0], lr=0.01333, weight_decay=0.0, betas=(0.9, 0.999))

    for epoch in range(3000):

        x = Variable(Tensor(shape=[1, 3], data=[[random.random(), random.random(), random.random()]]), name="x")
        y = VectorMatrixMultiply(Square(x), m0)  + b
        loss = Reduce(Square(y - (Square(x) * Constant(Tensor(shape=[1, 3], data=[[1.777, 1.333, 0.333]])) + Constant(Tensor(shape=[1, 3], data=[[1.55, 0.0, -1.666]]))))) # L2 loss to Ax^2+B

        print(f"Epoch {epoch}: loss = {loss.materialize().download()[0]}; a = {a.materialize().download()}, b = {b.materialize().download()}, \nm0 = \n{m0.materialize().download()}")
        # Backward pass
        # Gradient reset happens internally in the backward pass
        loss.backward()

        adamw.step()

    exit()

num_input_features = 64
num_nodes          = 64
batch_size         = 256
num_epochs         = 10000
m0 = Matrix(in_channels=num_input_features, out_features=num_nodes)
b0 = LearnableParameter(shape=[1, num_nodes,]) # bias
m1 = Matrix(in_channels=num_nodes, out_features=num_nodes)
b1 = LearnableParameter(shape=[1, num_nodes,]) # bias
m2 = Matrix(in_channels=num_nodes, out_features=num_nodes)
b2 = LearnableParameter(shape=[1, num_nodes]) # bias
m3 = Matrix(in_channels=num_nodes, out_features=3)
b3 = LearnableParameter(shape=[1, 3]) # bias

def broadcast_to_batch(node, batch_size):
    return Broadcast(node, dim=0, size=batch_size)

def broadcast_rgb_mul(k, batch_size):
    n = np.zeros((batch_size, 3), dtype=np.float32)
    n[:, 0] = k
    n[:, 1] = k
    n[:, 2] = k
    return n

def eval_mlp(x):
    batch_size = x.shape[0]
    z    = VectorMatrixMultiply(tensor=x, matrix=m0)
    z    = z + broadcast_to_batch(b0, batch_size)
    z    = LeakyRelu(z, negative_slope=0.1)
    z    = VectorMatrixMultiply(tensor=z, matrix=m1)
    z    = z + broadcast_to_batch(b1, batch_size)
    z    = LeakyRelu(z, negative_slope=0.1)
    z    = VectorMatrixMultiply(tensor=z, matrix=m2)
    z    = z + broadcast_to_batch(b2, batch_size)
    z    = LeakyRelu(z, negative_slope=0.1)
    z    = VectorMatrixMultiply(tensor=z, matrix=m3)
    # z    = z + broadcast_to_batch(b3, batch_size)
    # z    = LeakyRelu(z, negative_slope=0.01)
    return z

def eval_target(x):
    return x * x * 2.777 + 0.624 - x * x * x * 0.333 + np.sin(x * 5.0) * 0.777

adamw = AdamW(parameters=[m0, b0, m1, b1, m2, b2, m3, b3], lr=0.000533, weight_decay=0.01, betas=(0.92, 0.95))

import matplotlib.image as image
ref = image.imread("mlp_compression/assets/kodim23.png")[0:512, 0:512, 0:3]

assert ref.shape == (512, 512, 3), f"Unexpected reference image shape: {ref.shape}"

print(f"Reference image shape: {ref.shape}")

# Initialize all the feature vectors for every pixel on the image
size   = 512
x_data = np.zeros((size, size, num_input_features), dtype=np.float32)

print(f"Initializing the grid...")
# for pixel_x in range(size):
#     for pixel_y in range(size):
#         # Frequency encoding
#         for i in range(num_input_features // 4):
#             x_data[pixel_x, pixel_y, i * 4 + 0] = math.sin(pixel_x * (2.0 ** (i + 1)) * math.pi / size)
#             x_data[pixel_x, pixel_y, i * 4 + 1] = math.cos(pixel_x * (2.0 ** (i + 1)) * math.pi / size)
#             x_data[pixel_x, pixel_y, i * 4 + 2] = math.sin(pixel_y * (2.0 ** (i + 1)) * math.pi / size)
#             x_data[pixel_x, pixel_y, i * 4 + 3] = math.cos(pixel_y * (2.0 ** (i + 1)) * math.pi / size)

L      = num_input_features // 4
xx, yy = np.meshgrid(np.arange(size), np.arange(size))
freqs  = (2.0 ** (np.arange(L) + 1)) * np.pi / size
x_sin = np.sin(yy[:, :, None] * freqs[None, None, :])
x_cos = np.cos(yy[:, :, None] * freqs[None, None, :])
y_sin = np.sin(xx[:, :, None] * freqs[None, None, :])
y_cos = np.cos(xx[:, :, None] * freqs[None, None, :])
stacked = np.stack([x_sin, x_cos, y_sin, y_cos], axis=-1)
x_data[:] = stacked.reshape(size, size, -1)

print(f"Training...")

for epoch in range(num_epochs):
    _x_data = np.zeros((batch_size, num_input_features), dtype=np.float32)
    y_data = np.zeros((batch_size, 3), dtype=np.float32)

    for b in range(batch_size):
        pixel_x      = int(random.random() * ref.shape[0])
        pixel_y      = int(random.random() * ref.shape[1])
        _x_data[b]   = x_data[pixel_x, pixel_y]
        y_data[b, 0] = ref[pixel_x, pixel_y, 0]
        y_data[b, 1] = ref[pixel_x, pixel_y, 1]
        y_data[b, 2] = ref[pixel_x, pixel_y, 2]

    x    = Variable(Tensor(shape=[batch_size, num_input_features], data=_x_data), name="x")
    mlp  = eval_mlp(x)
    loss = Reduce(Square(mlp - Constant(Tensor(shape=[batch_size, 3], data=y_data))), op='+')

    if epoch == 0:
        with open(".tmp/graph.dot", "w") as f:
            f.write(loss.pretty_print_dot_graph())

    # Backward pass
    # Gradient reset happens internally in the backward pass
    loss.backward()

    adamw.step()

    queue.flush()

    print(f"Epoch {epoch}: loss = {loss.materialize().download()[0] / batch_size}; lr = {adamw.lr}")
    adamw.lr *= 0.99999

    # Poor mans garbage collection.
    # Only keep the leaf nodes of the graph
    # keep_alive_buffers = adamw.get_list_of_live_buffers()
    # for size in allocated_tensor_pool.keys():
    #     for buffer in allocated_tensor_pool[size]:
    #         if buffer not in keep_alive_buffers:
    #             free_buffer(buffer, size)
    

print(f"Allocated buffers: {get_number_of_allocated_buffers()}")
print(f"Free buffers: {get_number_of_free_buffers()}")

# Plot our mlp
import matplotlib.pyplot as plt


x_data = x_data.reshape((size * size, num_input_features))
x    = Variable(Tensor(shape=[size * size, num_input_features], data=x_data), name="x")
y_data = eval_mlp(x).materialize().download().reshape((size, size, 3))

plt.imshow(y_data, aspect="auto")
plt.axis("off")
plt.show()

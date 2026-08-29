"""
=============================================================================
LESSON 3 HANDS-ON: MEMORY COALESCING & SHARED MEMORY BANK CONFLICTS
=============================================================================
Run this on GPU (Google Colab or RTX 3090).

This script runs two real hardware experiments:
  1. Global Memory Coalescing (Stride 1 vs 2 vs 4 vs 8 vs 16 vs 32)
  2. Shared Memory Bank Conflicts (Conflict-free vs 2-way vs 32-way vs Padded)
=============================================================================
"""

import torch
import time
from torch.utils.cpp_extension import load_inline

cuda_source = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================================
// EXPERIMENT 1: GLOBAL MEMORY STRIDED ACCESS (Coalescing Test)
// ============================================================
// Stride 1 = Coalesced (Thread i reads element i)
// Stride S = Non-coalesced (Thread i reads element i * S)
// ============================================================================
__global__ void strided_read_kernel(const float* __restrict__ input, float* __restrict__ output, int N, int stride) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = (i * stride) % N;
    if (i < N) {
        output[i] = input[idx] * 2.0f;
    }
}

// ============================================================================
// EXPERIMENT 2: SHARED MEMORY BANK CONFLICTS
// ============================================================
// Shared memory has 32 banks (4 bytes wide each).
// bank_id = (address / 4) % 32
//
// Stride 1: Thread k accesses bank (k * 1) % 32 -> All 32 threads hit DIFFERENT banks (0 conflicts!)
// Stride 2: Thread k accesses bank (k * 2) % 32 -> 2 threads hit the SAME bank (2-way conflict)
// Stride 32: Thread k accesses bank (k * 32) % 32 -> All 32 threads hit Bank 0 (32-way conflict!)
// ============================================================================
__global__ void bank_conflict_kernel(float* output, int stride, int iterations) {
    __shared__ float s_mem[1024];

    int tid = threadIdx.x;
    s_mem[tid] = (float)tid;
    __syncthreads();

    float val = 0.0f;
    #pragma unroll 16
    for (int it = 0; it < iterations; ++it) {
        int index = (tid * stride + it) % 1024;
        val += s_mem[index];
    }

    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}

// ============================================================================
// EXPERIMENT 3: PADDED SHARED MEMORY (Fixing Bank Conflicts)
// ============================================================
// 2D Tile with standard float tile[32][32] -> Column access hits Bank 0 for all rows!
// 2D Tile with padded float tile[32][33]   -> Column access shifts by +1 bank each row (0 conflicts!)
// ============================================================================
__global__ void padded_vs_unpadded_kernel(float* output, bool use_padding, int iterations) {
    __shared__ float unpadded[32][32];
    __shared__ float padded[32][33]; // +1 padding column

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (use_padding) {
        padded[ty][tx] = (float)(ty * 33 + tx);
    } else {
        unpadded[ty][tx] = (float)(ty * 32 + tx);
    }
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll 16
    for (int it = 0; it < iterations; ++it) {
        // Access column-wise: thread tx accesses row 'k', col 'tx'
        int k = (it) % 32;
        if (use_padding) {
            sum += padded[k][tx];
        } else {
            sum += unpadded[k][tx];
        }
    }

    if (tx == 0 && ty == 0) {
        output[0] = sum;
    }
}

// ============================================================================
// PyTorch Bindings
// ============================================================
void run_strided_read(torch::Tensor input, torch::Tensor output, int stride) {
    int N = input.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    strided_read_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, stride
    );
}

void run_bank_conflict(torch::Tensor output, int stride, int iterations) {
    int blocks = 1000; // Enough blocks to saturate GPU
    int threads = 32;  // 1 warp
    bank_conflict_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        stride, iterations
    );
}

void run_padding_test(torch::Tensor output, bool use_padding, int iterations) {
    dim3 threads(32, 32); // 1024 threads
    int blocks = 500;
    padded_vs_unpadded_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        use_padding, iterations
    );
}
'''

cpp_source = '''
void run_strided_read(torch::Tensor input, torch::Tensor output, int stride);
void run_bank_conflict(torch::Tensor output, int stride, int iterations);
void run_padding_test(torch::Tensor output, bool use_padding, int iterations);
'''

print("=================================================================")
print("Compiling Lesson 3 CUDA Kernels with JIT nvcc...")
print("=================================================================")

module = load_inline(
    name='lesson3_memory',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['run_strided_read', 'run_bank_conflict', 'run_padding_test'],
    extra_cuda_cflags=['-O3'],
    verbose=False
)

print("✅ Kernel Compilation Succeeded!\n")


def benchmark(fn, *args, iters=50):
    # Warmup
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # ms


# =============================================================================
# RUN EXPERIMENT 1: Memory Coalescing
# =============================================================================
print("=" * 70)
print("EXPERIMENT 1: GLOBAL MEMORY COALESCING (Reading 64M floats = 256 MB)")
print("=" * 70)
N = 64 * 1024 * 1024  # 64 Million elements (256 MB)
inp = torch.randn(N, device='cuda', dtype=torch.float32)
out = torch.empty(N, device='cuda', dtype=torch.float32)

strides = [1, 2, 4, 8, 16, 32]
print(f"{'Stride':<10} | {'Time (ms)':<12} | {'Bandwidth (GB/s)':<18} | {'Status'}")
print("-" * 70)

base_bw = 0
for s in strides:
    t = benchmark(module.run_strided_read, inp, out, s)
    # Total data read + written: 256 MB read + 256 MB write = 512 MB
    bw = (N * 4 * 2) / (t * 1e-3) / (1024**3)
    if s == 1:
        base_bw = bw
        status = "✅ Coalesced (100% Efficiency)"
    else:
        drop = (1.0 - bw / base_bw) * 100
        status = f"⚠️ Strided ({drop:.1f}% Bandwidth Lost)"
    print(f"{s:<10} | {t:<12.3f} | {bw:<18.2f} | {status}")


# =============================================================================
# RUN EXPERIMENT 2: Shared Memory Bank Conflicts
# =============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 2: SHARED MEMORY BANK CONFLICTS (100,000 accesses/thread)")
print("=" * 70)
out_bank = torch.empty(1000, device='cuda', dtype=torch.float32)
iters = 100000

tests = [
    (1, "Stride 1: 0 Bank Conflicts (Each thread accesses Bank tid % 32)", "✅ Perfect"),
    (2, "Stride 2: 2-Way Conflict   (Threads 0 & 16 hit Bank 0)", "⚠️ 2x Slower"),
    (4, "Stride 4: 4-Way Conflict   (4 threads hit same Bank)", "⚠️ 4x Slower"),
    (8, "Stride 8: 8-Way Conflict   (8 threads hit same Bank)", "🔴 8x Slower"),
    (16, "Stride 16: 16-Way Conflict (16 threads hit same Bank)", "🔴 16x Slower"),
    (32, "Stride 32: 32-Way Conflict (ALL 32 threads hit Bank 0!)", "🔴 Serialized"),
]

print(f"{'Pattern':<58} | {'Time (ms)':<10} | {'Slowdown'}")
print("-" * 80)

base_time = 0
for stride, desc, status in tests:
    t = benchmark(module.run_bank_conflict, out_bank, stride, iters)
    if stride == 1:
        base_time = t
        slowdown = "1.00x (Baseline)"
    else:
        factor = t / base_time
        slowdown = f"{factor:.2f}x slower"
    print(f"{desc:<58} | {t:<10.3f} | {slowdown}")


# =============================================================================
# RUN EXPERIMENT 3: Padded vs Unpadded Shared Memory Tile
# =============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: COLUMN ACCESS IN 2D TILE (Padding Fix)")
print("=" * 70)
out_pad = torch.empty(1, device='cuda', dtype=torch.float32)

t_unpadded = benchmark(module.run_padding_test, out_pad, False, 50000)
t_padded = benchmark(module.run_padding_test, out_pad, True, 50000)

print(f"Unpadded tile[32][32] (32-way Bank Conflict on Col Reads): {t_unpadded:.3f} ms")
print(f"Padded   tile[32][33] (+1 Column Shift, 0 Conflicts):     {t_padded:.3f} ms")
print(f"✅ Speedup from adding 1 dummy column padding:             {t_unpadded / t_padded:.2f}x faster!")
print("=" * 70)

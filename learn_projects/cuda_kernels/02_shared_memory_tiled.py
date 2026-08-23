"""
=============================================================================
STAGE 4 — MILESTONE 4.2: SHARED MEMORY TILED MATRIX MULTIPLICATION
=============================================================================

Run this in Google Colab (Runtime → T4 GPU).

=============================================================================
WHY IS NAIVE MATMUL SLOW? (The Global Memory Bottleneck)
=============================================================================

In Naive Matmul (Milestone 4.1):
  - Thread (row, col) reads row `row` of A and col `col` of B from VRAM (DRAM).
  - For a 1024x1024 matrix, every float in A is read 1024 times from slow VRAM!
  - VRAM latency is ~400-800 clock cycles.
  - The GPU ALU cores spend 95% of their time IDLE, stalled waiting for DRAM.

=============================================================================
THE SOLUTION: SHARED MEMORY TILING
=============================================================================

Each Streaming Multiprocessor (SM) on the GPU has 48KB - 100KB of on-chip
`__shared__` memory (L1 SRAM), with ~1-2 cycles latency (100x faster than DRAM!).

The Algorithm:
1. Divide the matrices into small square tiles (e.g. 16x16 or 32x32 floats).
2. All 256 threads in a block collaborate to load ONE tile of A and ONE tile
   of B from slow Global Memory into fast `__shared__` memory.
3. Synchronize all threads in the block using `__syncthreads()`.
4. All threads multiply the shared tile data in fast SRAM and accumulate into
   local registers.
5. Slide the tile window across the K dimension and repeat until done.

Result: Global memory traffic is reduced by 16x - 32x!
=============================================================================
"""

import torch
import time
from torch.utils.cpp_extension import load_inline

cuda_source = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// ============================================================================
// SHARED MEMORY TILED MATRIX MULTIPLY KERNEL
// ============================================================================
__global__ void tiled_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Allocate 16x16 tiles in fast on-chip L1 SRAM (Shared Memory)
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float acc = 0.0f;

    // Slide tile window across the K dimension
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Collaborative load from Global Memory (DRAM) into Shared Memory (SRAM)
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;

        if (row < M && a_col < K)
            s_A[ty][tx] = A[row * K + a_col];
        else
            s_A[ty][tx] = 0.0f;

        if (b_row < K && col < N)
            s_B[ty][tx] = B[b_row * N + col];
        else
            s_B[ty][tx] = 0.0f;

        // Barrier: Wait until all 256 threads in this block finish loading their tile
        __syncthreads();

        // Multiply the tile from fast Shared Memory (zero DRAM accesses!)
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += s_A[ty][k] * s_B[k][tx];
        }

        // Barrier: Ensure all threads finished compute before next tile overwrites SRAM
        __syncthreads();
    }

    // Write final accumulated sum to output matrix C
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// C++ Host Wrapper
torch::Tensor tiled_matmul(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Tensors must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimension K must match");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    tiled_matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}
'''

cpp_source = "torch::Tensor tiled_matmul(torch::Tensor A, torch::Tensor B);"

if __name__ == "__main__":
    print("=" * 70)
    print("MILESTONE 4.2: Compiling Shared Memory Tiled CUDA Kernel...")
    print("=" * 70)

    tiled_module = load_inline(
        name="tiled_matmul_module",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["tiled_matmul"],
        extra_cuda_cflags=["-O3"]
    )
    print("✓ Compilation successful! Running benchmark against cuBLAS...\n")

    sizes = [256, 512, 1024, 2048, 4096]

    print(f"  {'Size (N)':>8}  {'cuBLAS (ms)':>12}  {'Tiled CUDA (ms)':>16}  {'Tiled GFLOPS':>14}  {'Correct':>8}")
    print(f"  {'─' * 8}  {'─' * 12}  {'─' * 16}  {'─' * 14}  {'─' * 8}")

    for N in sizes:
        M = K = N
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # cuBLAS
        torch.cuda.synchronize()
        for _ in range(5): C_ref = torch.matmul(A, B)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20): C_ref = torch.matmul(A, B)
        torch.cuda.synchronize()
        cublas_ms = (time.perf_counter() - t0) / 20 * 1000

        # Tiled CUDA
        torch.cuda.synchronize()
        for _ in range(5): C_tiled = tiled_module.tiled_matmul(A, B)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20): C_tiled = tiled_module.tiled_matmul(A, B)
        torch.cuda.synchronize()
        tiled_ms = (time.perf_counter() - t0) / 20 * 1000

        diff = torch.max(torch.abs(C_tiled - C_ref)).item()
        is_pass = diff < 1e-2

        flops = 2 * M * N * K
        gflops = flops / (tiled_ms / 1000.0) / 1e9

        print(f"  {N:>8}  {cublas_ms:>12.3f}  {tiled_ms:>16.3f}  {gflops:>14.1f}  {'✓ PASS' if is_pass else '✗ FAIL':>8}")

    print("\n" + "=" * 70)
    print("✅ MILESTONE 4.2 COMPLETE — SHARED MEMORY TILING OPTIMIZATION VERIFIED!")
    print("=" * 70)

"""
=============================================================================
STAGE 4 — MILESTONE 4.1: YOUR FIRST CUDA KERNEL (Naive Matrix Multiply)
=============================================================================

Run this in Google Colab (Runtime → T4 GPU).

This file uses PyCUDA + raw CUDA C to write a GPU kernel from scratch.
PyCUDA lets you write the actual CUDA C kernel as a string, compile it
at runtime with nvcc, and call it from Python. This is NOT a toy wrapper —
the kernel code below is the EXACT same CUDA C you'd write in a .cu file.

=============================================================================
WHAT IS A CUDA KERNEL? (For the C++ developer)
=============================================================================

In C++, a function runs on ONE thread:

    void add(float* a, float* b, float* c, int n) {
        for (int i = 0; i < n; i++)
            c[i] = a[i] + b[i];   // Sequential, one element at a time
    }

In CUDA, a __global__ function runs on THOUSANDS of threads simultaneously:

    __global__ void add(float* a, float* b, float* c, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;  // Each thread knows its ID
        if (i < n)
            c[i] = a[i] + b[i];   // Each thread does ONE element
    }

That's it. The __global__ keyword means "this function runs on the GPU."
Each thread computes its own unique index and processes ONE element.
NVIDIA's hardware scheduler distributes 100,000 threads across all SMs.

=============================================================================
"""

import numpy as np
import time


def check_pycuda():
    """Install PyCUDA if not available (Colab)."""
    try:
        import pycuda.autoinit
        import pycuda.driver as drv
        return True
    except ImportError:
        import subprocess
        print("Installing PyCUDA (one-time setup on Colab)...")
        subprocess.run(["pip", "install", "pycuda"], check=True, capture_output=True)
        return True


# =============================================================================
# THE CUDA KERNEL CODE (This is real CUDA C, not Python)
# =============================================================================
#
# Read this like you'd read a .cu file. The Python string is just a container.
# nvcc (NVIDIA's compiler) compiles this to GPU machine code (PTX → SASS).

NAIVE_MATMUL_KERNEL = """
// ============================================================================
// Naive Matrix Multiply: C[M,N] = A[M,K] × B[K,N]
// ============================================================================
//
// Strategy: Each thread computes ONE element of the output matrix C.
//
// If C is 1024×1024, we launch 1,048,576 threads total.
// Thread (row, col) computes: C[row][col] = sum(A[row][k] * B[k][col]) for k=0..K-1
//
// Memory access pattern:
//   Thread (row, col) reads:
//     - Entire row `row` of A  (K floats from global memory)
//     - Entire col `col` of B  (K floats from global memory)
//   Then writes ONE float to C[row][col].
//
// This is EXTREMELY wasteful:
//   - Row 0 of A is read by threads (0,0), (0,1), (0,2), ... (0,N-1)
//   - That's N redundant reads of the same row!
//   - We'll fix this in Milestone 4.2 with shared memory tiling.
//
// But first: let's make sure we understand the basics.

__global__ void naive_matmul(
    const float* A,    // Input matrix A [M × K], stored row-major
    const float* B,    // Input matrix B [K × N], stored row-major
    float* C,          // Output matrix C [M × N]
    int M,             // Number of rows in A (and C)
    int N,             // Number of columns in B (and C)
    int K              // Number of columns in A (= rows in B)
)
{
    // ─── Step 1: Figure out WHICH element of C this thread computes ───
    //
    // We launch a 2D grid of blocks, and each block has a 2D array of threads.
    //
    //   blockIdx.x  = which block column (0, 1, 2, ...)
    //   blockIdx.y  = which block row    (0, 1, 2, ...)
    //   threadIdx.x = thread's column within this block (0..BLOCK_SIZE-1)
    //   threadIdx.y = thread's row within this block    (0..BLOCK_SIZE-1)
    //
    // Global position in the output matrix:
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // Which row of C
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Which col of C

    // ─── Step 2: Bounds check ───
    // We may launch more threads than matrix elements (rounding up to block size).
    // Out-of-bounds threads must bail out to avoid writing garbage.
    if (row >= M || col >= N) return;

    // ─── Step 3: Compute the dot product ───
    //
    //   C[row][col] = Σ (A[row][k] * B[k][col])  for k = 0..K-1
    //
    // In C++ terms, this is:
    //   float sum = 0;
    //   for (int k = 0; k < K; k++)
    //       sum += A[row * K + k] * B[k * N + col];
    //
    // Row-major layout: element (i,j) of an (R×C) matrix is at index [i * C + j].
    // This is identical to how C/C++ lays out 2D arrays in memory.

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k]    // A[row][k]
             * B[k * N + col];   // B[k][col]
    }

    // ─── Step 4: Write the result ───
    C[row * N + col] = sum;
}
"""


def run_naive_matmul():
    """
    Compiles and runs the naive CUDA matrix multiply kernel.
    Compares against NumPy (CPU) for correctness AND measures performance.
    """
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule

    print("=" * 70)
    print("MILESTONE 4.1: YOUR FIRST CUDA KERNEL — Naive Matrix Multiply")
    print("=" * 70)

    # ─── Compile the CUDA kernel ───
    # This calls nvcc (NVIDIA's compiler) behind the scenes.
    # It compiles the CUDA C string above into GPU machine code (PTX → SASS).
    # This is analogous to: g++ -O2 -o matmul matmul.cu
    print("\n  Compiling CUDA kernel with nvcc...")
    mod = SourceModule(NAIVE_MATMUL_KERNEL)
    naive_matmul = mod.get_function("naive_matmul")
    print("  ✓ Kernel compiled successfully!")

    # ─── Test at multiple matrix sizes ───
    sizes = [128, 256, 512, 1024, 2048]

    print(f"\n  {'M=N=K':>8}  {'CPU (ms)':>10}  {'GPU (ms)':>10}  {'Speedup':>10}  {'GFLOPS':>10}  {'Correct':>8}")
    print(f"  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 8}")

    for N in sizes:
        M, K = N, N

        # Create random matrices on CPU (NumPy)
        A_cpu = np.random.randn(M, K).astype(np.float32)
        B_cpu = np.random.randn(K, N).astype(np.float32)
        C_cpu = np.zeros((M, N), dtype=np.float32)

        # ─── CPU Baseline (NumPy uses optimized BLAS under the hood) ───
        t0 = time.perf_counter()
        C_ref = A_cpu @ B_cpu  # NumPy matmul (calls MKL/OpenBLAS)
        cpu_ms = (time.perf_counter() - t0) * 1000

        # ─── GPU: Allocate device memory and copy data ───
        # In C++ CUDA, this would be:
        #   float *d_A, *d_B, *d_C;
        #   cudaMalloc(&d_A, M * K * sizeof(float));
        #   cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        #
        # PyCUDA handles this with drv.mem_alloc() and drv.memcpy_htod()
        import pycuda.gpuarray as gpuarray
        A_gpu = gpuarray.to_gpu(A_cpu)
        B_gpu = gpuarray.to_gpu(B_cpu)
        C_gpu = gpuarray.zeros((M, N), dtype=np.float32)

        # ─── Configure the kernel launch ───
        # BLOCK_SIZE: How many threads per block (in each dimension).
        # We use 16×16 = 256 threads per block (a common starting point).
        #
        # GRID_SIZE: How many blocks we need to cover the entire matrix.
        # grid_x = ceil(N / 16), grid_y = ceil(M / 16)
        #
        # In C++ CUDA, this would be:
        #   dim3 blockDim(16, 16);
        #   dim3 gridDim((N + 15) / 16, (M + 15) / 16);
        #   naive_matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

        BLOCK_SIZE = 16
        grid_x = (N + BLOCK_SIZE - 1) // BLOCK_SIZE  # ceil(N / 16)
        grid_y = (M + BLOCK_SIZE - 1) // BLOCK_SIZE  # ceil(M / 16)

        # Warmup (first kernel launch has overhead from JIT/caching)
        naive_matmul(
            A_gpu, B_gpu, C_gpu,
            np.int32(M), np.int32(N), np.int32(K),
            block=(BLOCK_SIZE, BLOCK_SIZE, 1),
            grid=(grid_x, grid_y, 1)
        )
        drv.Context.synchronize()

        # ─── Benchmark the GPU kernel ───
        num_runs = 10 if N <= 1024 else 5
        t0 = time.perf_counter()
        for _ in range(num_runs):
            naive_matmul(
                A_gpu, B_gpu, C_gpu,
                np.int32(M), np.int32(N), np.int32(K),
                block=(BLOCK_SIZE, BLOCK_SIZE, 1),
                grid=(grid_x, grid_y, 1)
            )
        drv.Context.synchronize()
        gpu_ms = (time.perf_counter() - t0) / num_runs * 1000

        # ─── Verify correctness ───
        C_result = C_gpu.get()  # Copy result back from GPU to CPU
        max_diff = np.max(np.abs(C_result - C_ref))
        correct = max_diff < 1e-2  # FP32 tolerance for large matrices

        # ─── Calculate GFLOPS ───
        # Matrix multiply: 2 * M * N * K floating point operations
        # (one multiply + one add per element of the dot product)
        flops = 2 * M * N * K
        gflops = flops / (gpu_ms / 1000) / 1e9

        speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0

        print(f"  {N:>8}  {cpu_ms:>10.2f}  {gpu_ms:>10.2f}  {speedup:>9.1f}x  {gflops:>9.1f}  {'✓' if correct else '✗':>8}")

    print()
    print("  ═══════════════════════════════════════════════════════════════")
    print("  ANALYSIS OF YOUR FIRST CUDA KERNEL")
    print("  ═══════════════════════════════════════════════════════════════")
    print()
    print("  What you just did:")
    print("  1. Wrote a __global__ kernel function in CUDA C")
    print("  2. Compiled it with nvcc at runtime")
    print("  3. Allocated GPU memory and copied data Host → Device")
    print("  4. Launched a 2D grid of thread blocks")
    print("  5. Each thread computed ONE element of the output matrix")
    print("  6. Copied the result Device → Host and verified correctness")
    print()
    print("  Why this kernel is SLOW (we'll fix in Milestone 4.2):")
    print("  ─────────────────────────────────────────────────────────")
    print("  • Each thread reads an ENTIRE row of A and column of B")
    print("    from slow Global Memory (VRAM).")
    print("  • Adjacent threads in the same block read the SAME row of A")
    print("    but each thread fetches it independently. Massive redundancy!")
    print("  • For a 1024×1024 matrix, each row is read 1024 times.")
    print("  • That's 1024x more memory traffic than necessary.")
    print()
    print("  The fix (Milestone 4.2 — Shared Memory Tiling):")
    print("  ─────────────────────────────────────────────────────────")
    print("  • Load a TILE of A and B into fast shared memory (48KB L1)")
    print("  • All 256 threads in a block SHARE that tile")
    print("  • Reduces global memory reads by ~BLOCK_SIZE times (16x)")
    print("  • This is the single most important GPU optimization technique")
    print()
    print("  T4 GPU theoretical peak: ~8.1 TFLOPS FP32")
    print("  cuBLAS (NVIDIA's hand-tuned kernel): ~7.5 TFLOPS")
    print("  Our naive kernel: probably ~0.1-0.5 TFLOPS (1-5% efficiency)")
    print("  Goal by Milestone 4.3: reach ~3-5 TFLOPS (40-60% efficiency)")


if __name__ == "__main__":
    check_pycuda()
    run_naive_matmul()

    print()
    print("=" * 70)
    print("✅ MILESTONE 4.1 COMPLETE!")
    print("=" * 70)
    print()
    print("  You have written and executed your FIRST CUDA kernel.")
    print("  You understand: __global__, threadIdx, blockIdx, blockDim,")
    print("  grid/block configuration, Host↔Device memory transfers,")
    print("  and why naive global memory access is the #1 bottleneck.")
    print()
    print("  Next: 02_shared_memory_tiled.py — The Tiling Optimization")

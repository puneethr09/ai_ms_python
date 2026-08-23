"""
=============================================================================
STAGE 4 — MILESTONE 0: GPU FUNDAMENTALS FOR THE C++ ENGINEER
=============================================================================

Run this in Google Colab (Runtime → T4 GPU) to verify your GPU is working
and to understand the hardware you're programming against.

=============================================================================
THE BIG PICTURE: WHY GPUs EXIST
=============================================================================

In C++, you think about computation like this:

    for (int i = 0; i < 1000000; i++) {
        result[i] = a[i] * b[i];  // One core does this sequentially
    }

    // With threading (8 cores):
    // Thread 0 handles i = 0..124999
    // Thread 1 handles i = 125000..249999
    // ...
    // Thread 7 handles i = 875000..999999

A CPU core is a **Swiss Army knife**: branch prediction, out-of-order execution,
large caches, speculative execution. It can do ANYTHING fast.

A GPU core is a **hammer**: it can only do simple math, but NVIDIA gives you
**10,000 hammers running at the same time**.

    // GPU way (conceptually):
    // 10,000+ threads ALL execute simultaneously:
    // Thread 0:    result[0]    = a[0]    * b[0]
    // Thread 1:    result[1]    = a[1]    * b[1]
    // Thread 2:    result[2]    = a[2]    * b[2]
    // ...
    // Thread 9999: result[9999] = a[9999] * b[9999]

=============================================================================
THE GPU HARDWARE MODEL (Map to what you know)
=============================================================================

CPU World (what you know)          →    GPU World (what you're learning)
─────────────────────────────────       ─────────────────────────────────
Machine (your desktop)             →    Device (the GPU card)
CPU cores (8-16)                   →    Streaming Multiprocessors "SMs" (40-144)
Hyper-threads per core (2)         →    CUDA Cores per SM (64-128)
L1 Cache (per core, 64KB)         →    Shared Memory (per SM, 48-100KB)
L2 Cache (shared, 8-32MB)         →    L2 Cache (shared, 6-72MB)
RAM / DRAM (32-128GB)             →    Global Memory / VRAM (8-80GB)
std::thread                        →    CUDA Thread
                                        (but you launch 100,000+ of them)

Key difference: A CPU core is smart (branch prediction, OOO execution).
               A GPU core is dumb (just does math) but there are THOUSANDS.

=============================================================================
THE CUDA THREAD HIERARCHY (The most important concept)
=============================================================================

In C++, you create threads manually:
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; i++)
        threads.emplace_back(worker, i);

In CUDA, you describe a GRID of threads using a 3-level hierarchy:

    ┌──────────────────────────────────────────────────────────────────┐
    │                         GRID                                     │
    │  (The entire launch. Think: the "main" dispatch)                 │
    │                                                                  │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
    │  │ Block    │ │ Block    │ │ Block    │ │ Block    │  ...       │
    │  │ (0,0)    │ │ (1,0)    │ │ (2,0)    │ │ (3,0)    │           │
    │  │          │ │          │ │          │ │          │           │
    │  │ 256      │ │ 256      │ │ 256      │ │ 256      │           │
    │  │ threads  │ │ threads  │ │ threads  │ │ threads  │           │
    │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
    │  │ Block    │ │ Block    │ │ Block    │ │ Block    │  ...       │
    │  │ (0,1)    │ │ (1,1)    │ │ (2,1)    │ │ (3,1)    │           │
    │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
    └──────────────────────────────────────────────────────────────────┘

    Key rules:
    1. A GRID contains many BLOCKS.
    2. A BLOCK contains many THREADS (typically 128 or 256).
    3. All threads in a BLOCK can share fast "shared memory" (like L1 cache).
    4. Threads in DIFFERENT blocks CANNOT communicate directly.
    5. Within a block, threads execute in groups of 32 called "WARPS"
       (think: SIMD lanes — all 32 threads execute the SAME instruction).

    Every thread knows its own identity:
    - threadIdx.x  → which thread am I within my block? (0..255)
    - blockIdx.x   → which block am I in? (0..N)
    - blockDim.x   → how many threads per block? (e.g., 256)
    - gridDim.x    → how many blocks total?

    So thread's global index = blockIdx.x * blockDim.x + threadIdx.x

=============================================================================
THE GPU MEMORY HIERARCHY (This is where performance lives or dies)
=============================================================================

    Speed       Memory Type           Size        Scope          C++ Analogy
    ─────────   ───────────────────   ─────────   ────────────   ────────────
    FASTEST     Registers             ~256KB/SM   Per thread     CPU registers
    ↑           Shared Memory         48-100KB    Per block      L1 cache (but you control it)
    │           L2 Cache              6-72MB      All SMs        L2 cache
    ↓           Global Memory (VRAM)  8-80GB      All threads    DRAM / main RAM
    SLOWEST

    The golden rule: Global memory access is ~100x slower than shared memory.
    The entire art of CUDA optimization is: move data from Global → Shared → Registers,
    compute on it, then write results back to Global.

    This is EXACTLY like CPU cache optimization, except on GPU you MANUALLY
    control what goes into "L1" (shared memory). On CPU, the hardware cache
    controller does this for you automatically.

=============================================================================
"""

import subprocess
import sys


def run_nvidia_smi():
    """Show the GPU hardware available in this Colab session."""
    print("=" * 70)
    print("STEP 1: Verify GPU Hardware")
    print("=" * 70)
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: No NVIDIA GPU detected!")
        print("In Google Colab: Runtime → Change runtime type → T4 GPU")
        sys.exit(1)
    print(result.stdout)


def query_gpu_properties():
    """Query detailed GPU properties using PyTorch."""
    print("=" * 70)
    print("STEP 2: GPU Properties (Your Hardware Spec Sheet)")
    print("=" * 70)

    import torch

    if not torch.cuda.is_available():
        print("CUDA not available! Check GPU runtime.")
        return

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    print(f"  GPU Name:                {props.name}")
    print(f"  Compute Capability:      {props.major}.{props.minor}")
    print(f"  Total VRAM (Global Mem): {props.total_mem / 1e9:.1f} GB")
    print(f"  Streaming Multiprocessors (SMs): {props.multi_processor_count}")
    print(f"  Max Threads per Block:   {props.max_threads_per_block}")
    print(f"  Max Block Dimensions:    {props.max_threads_per_block}")
    print(f"  Warp Size:               {props.warp_size}")
    print()

    # Calculate theoretical peak
    # T4: 2560 CUDA cores @ 1.59 GHz = ~8.1 TFLOPS FP32
    # RTX 3090: 10496 CUDA cores @ 1.70 GHz = ~35.6 TFLOPS FP32
    cuda_cores_per_sm = {
        "Tesla T4": 64,
        "NVIDIA A100": 64,
        "NVIDIA H100": 128,
        "NVIDIA GeForce RTX 3090": 128,
        "NVIDIA GeForce RTX 4090": 128,
    }
    cores_per = cuda_cores_per_sm.get(props.name, 64)
    total_cores = props.multi_processor_count * cores_per

    print(f"  === DERIVED SPECS ===")
    print(f"  Estimated CUDA Cores:    {total_cores}")
    print(f"  (= {props.multi_processor_count} SMs × {cores_per} cores/SM)")
    print()
    print(f"  Think of it as: {props.multi_processor_count} independent \"CPUs\",")
    print(f"  each with {cores_per} ALUs that execute in lockstep (SIMD-like).")
    print(f"  Total parallel compute units: {total_cores}")


def demonstrate_cpu_vs_gpu_throughput():
    """
    The 'aha moment': demonstrate that GPU is slower for small work
    but MASSIVELY faster for large parallel work.
    """
    print()
    print("=" * 70)
    print("STEP 3: CPU vs GPU — When GPUs Win (and When They Lose)")
    print("=" * 70)

    import torch
    import time

    sizes = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]

    print(f"\n  {'Vector Size':>12}  {'CPU (ms)':>10}  {'GPU (ms)':>10}  {'Speedup':>10}  {'Winner':>8}")
    print(f"  {'─' * 12}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 8}")

    for N in sizes:
        # CPU
        a_cpu = torch.randn(N)
        b_cpu = torch.randn(N)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(100):
            c_cpu = a_cpu * b_cpu
        cpu_ms = (time.perf_counter() - t0) / 100 * 1000

        # GPU
        a_gpu = torch.randn(N, device="cuda")
        b_gpu = torch.randn(N, device="cuda")
        torch.cuda.synchronize()

        # Warmup
        for _ in range(10):
            c_gpu = a_gpu * b_gpu
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(100):
            c_gpu = a_gpu * b_gpu
        torch.cuda.synchronize()
        gpu_ms = (time.perf_counter() - t0) / 100 * 1000

        speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0
        winner = "GPU" if speedup > 1.0 else "CPU"

        print(f"  {N:>12,}  {cpu_ms:>10.3f}  {gpu_ms:>10.3f}  {speedup:>9.1f}x  {winner:>8}")

    print()
    print("  KEY INSIGHT:")
    print("  ─────────────────────────────────────────────────────────────")
    print("  Small vectors: CPU wins (GPU has ~5μs kernel launch overhead).")
    print("  Large vectors: GPU wins by 10-100x (thousands of cores in parallel).")
    print("  Neural networks deal with MASSIVE tensors → GPU always wins.")
    print()
    print("  This is why LLM inference is GPU-bound: a single forward pass")
    print("  of a 7B parameter model does ~14 billion multiply-adds.")


def demonstrate_memory_bandwidth():
    """
    Show that GPU performance is almost always LIMITED by memory bandwidth,
    not compute. This is the single most important insight for optimization.
    """
    print()
    print("=" * 70)
    print("STEP 4: The Memory Wall (Why Optimization = Memory Optimization)")
    print("=" * 70)

    import torch
    import time

    N = 4096
    A = torch.randn(N, N, device="cuda")
    B = torch.randn(N, N, device="cuda")

    # Warmup
    for _ in range(3):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(10):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / 10

    # Calculate FLOPS
    # Matrix multiply C = A × B where A is (M,K) and B is (K,N)
    # requires 2*M*N*K floating point operations (multiply + add)
    flops = 2 * N * N * N
    gflops = flops / elapsed / 1e9
    tflops = gflops / 1000

    bytes_read = (N * N + N * N) * 4  # Two matrices, float32 = 4 bytes
    bytes_written = N * N * 4
    total_bytes = bytes_read + bytes_written
    bandwidth_gbps = total_bytes / elapsed / 1e9

    print(f"\n  Matrix Multiply: ({N} × {N}) @ ({N} × {N})")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Time:            {elapsed * 1000:.2f} ms")
    print(f"  Compute:         {tflops:.2f} TFLOPS")
    print(f"  Memory Traffic:  {total_bytes / 1e6:.0f} MB")
    print(f"  Bandwidth Used:  {bandwidth_gbps:.1f} GB/s")
    print()
    print(f"  FLOP-to-Byte Ratio: {flops / total_bytes:.0f} : 1")
    print()
    print("  KEY INSIGHT:")
    print("  ─────────────────────────────────────────────────────────────")
    print("  Matrix multiply has a HIGH compute-to-memory ratio (~2N:1).")
    print("  That's why GPUs love it — they can keep their cores busy")
    print("  while waiting for memory. This ratio is called the")
    print("  'arithmetic intensity' and it's the key to the ROOFLINE MODEL.")
    print()
    print("  When we write our OWN CUDA kernels in the next milestone,")
    print("  the entire game is maximizing this ratio by reusing data")
    print("  in fast shared memory instead of re-reading from slow VRAM.")


def explain_matmul_is_everything():
    """Show that neural network inference is 95%+ matrix multiply."""
    print()
    print("=" * 70)
    print("STEP 5: Why We're Building Matrix Multiply Kernels")
    print("=" * 70)
    print("""
  Every layer of a neural network (including LLMs) is fundamentally:

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  Linear Layer:    output = input × weights + bias               │
    │                           ^^^^^^^^^^^^^^                        │
    │                           THIS IS MATMUL                        │
    │                                                                 │
    │  Attention:       Q = X × W_q                    ← matmul      │
    │                   K = X × W_k                    ← matmul      │
    │                   V = X × W_v                    ← matmul      │
    │                   scores = Q × K^T               ← matmul      │
    │                   output = softmax(scores) × V   ← matmul      │
    │                   projected = output × W_o       ← matmul      │
    │                                                                 │
    │  Feed-Forward:    hidden = input × W_up          ← matmul      │
    │                   output = activation(hidden) × W_down ← matmul│
    │                                                                 │
    │  That's 8 matmuls PER LAYER. A 32-layer LLM does 256 matmuls   │
    │  PER TOKEN. Optimizing matmul = optimizing ALL of AI inference. │
    └─────────────────────────────────────────────────────────────────┘

  In Stage 2, you wrote the C++ version of this (matrix-vector multiply
  in your custom_infer.cpp). Now we're writing the GPU-parallel version
  that runs 100-1000x faster.

  In the NEXT milestone (01_naive_matmul.py), you will write your FIRST
  __global__ CUDA kernel — a raw matrix multiply that runs on the GPU.
""")


if __name__ == "__main__":
    run_nvidia_smi()
    query_gpu_properties()
    demonstrate_cpu_vs_gpu_throughput()
    demonstrate_memory_bandwidth()
    explain_matmul_is_everything()

    print()
    print("=" * 70)
    print("✅ MILESTONE 0 COMPLETE!")
    print("=" * 70)
    print()
    print("You now understand:")
    print("  1. GPU hardware: SMs, CUDA cores, warps")
    print("  2. Thread hierarchy: Grid → Blocks → Threads")
    print("  3. Memory hierarchy: Registers → Shared → L2 → Global (VRAM)")
    print("  4. When GPUs win vs CPUs (large parallel workloads)")
    print("  5. The Memory Wall (optimization = memory access optimization)")
    print("  6. Why matmul is the only kernel that matters for AI")
    print()
    print("Next: 01_naive_matmul.py — Your first CUDA kernel!")

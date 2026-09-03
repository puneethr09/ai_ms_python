# ⚡ Stage 4: GPU Programming & Custom CUDA Kernels
> **From SIMT Warps & Hardware Schedulers to High-Throughput Matrix Multiply Tiling & Memory Coalescing**

---

## 🧠 The Mental Model Shift (CPU vs. GPU)

You've spent your career writing code that runs on **8–16 fast, hyper-optimized cores** (CPU).  
A modern GPU has **thousands of simpler cores** that all execute the **same instruction** across 32-thread warps simultaneously (SIMT - Single Instruction, Multiple Threads).

| Dimension | CPU (Host) | GPU (Device - Tesla T4) |
| :--- | :--- | :--- |
| **Core Architecture** | 8–16 Out-of-Order speculative cores | 40 Streaming Multiprocessors (SMs) = 2,560 CUDA Cores |
| **Clock Speed** | 4.0 – 5.5 GHz | 1.4 – 1.7 GHz (Lower to prevent thermal runaway across 2,560+ cores) |
| **Latency Strategy** | Massive caches + branch predictors to minimize latency (~60 ns) | **Latency Hiding:** Zero-cost warp context switching across 400-cycle DRAM stalls |
| **Memory Bandwidth** | 50 – 100 GB/s (DDR4/DDR5) | 320 GB/s (GDDR6) to 3,350+ GB/s (HBM3) |
| **Execution Model** | Independent threads with individual Program Counters | 32 threads in lockstep (**Warp**) sharing 1 instruction dispatcher |

---

## 📁 Project Structure & Milestones

```
cuda_kernels/
├── README.md                                  ← You are here
├── CUDA_Masterclass.ipynb                     ← Interactive Colab / GPU Masterclass Notebook
├── LESSON_1_GPU_HARDWARE_FUNDAMENTALS.md      ← Lesson 1: Silicon layout, SMs, Warps & Memory Bus
├── LESSON_2_CUDA_PROGRAMMING_MODEL.md         ← Lesson 2: 1D & 2D Grid/Block coordinates & GigaThread
├── LESSON_3_MEMORY_HIERARCHY_AND_BENCHMARKS.md← Lesson 3: Caching, Allocators, Coalescing & Bank Conflicts
│
├── 00_gpu_fundamentals.py                     ← Milestone 0: Hardware topology & CPU vs GPU crossover
├── 01_naive_matmul.py                         ← Milestone 1: First raw CUDA C++ kernel (JIT-compiled)
├── 02_shared_memory_tiled.py                  ← Milestone 2: On-chip L1 SRAM tiled GEMM (3.05x speedup)
├── 03_coalescing_and_bank_conflicts.py        ← Milestone 3: Memory bus coalescing & 32 bank conflict tests
│
├── colab_runner.py                            ← Automated JIT benchmark runner for Google Colab
├── colab_all_in_one.py                        ← Self-contained standalone compilation script
└── run_benchmarks_direct.py                   ← Direct execution script
```

---

## 📖 Theoretical Foundations (Complete Reference Guides)

| Guide | Core Hardware & Systems Insights |
| :--- | :--- |
| ⚡ [**Lesson 1: GPU Hardware Fundamentals**](LESSON_1_GPU_HARDWARE_FUNDAMENTALS.md) | SM internals, Turing vs Ampere Tensor Cores, SIMT lockstep, branch divergence penalties, zero-cost context switching, and GDDR6 vs HBM packaging. |
| 🧩 [**Lesson 2: CUDA Programming Model**](LESSON_2_CUDA_PROGRAMMING_MODEL.md) | Replacing loops with thread IDs, Grid $\to$ Block $\to$ Warp hierarchy, 1D & 2D coordinate formulas (`col = blockIdx.x * blockDim.x + threadIdx.x`), and GigaThread SM scheduling. |
| 🔬 [**Lesson 3: Memory Hierarchy & Benchmarks**](LESSON_3_MEMORY_HIERARCHY_AND_BENCHMARKS.md) | Asynchronous CUDA streams & event timing, `cudaMalloc` (0.5–5 ms) driver overhead vs PyTorch Caching Allocator, DRAM coalesced transactions, and shared memory bank conflicts. |

---

## 📊 Live Silicon Benchmarks (Verified on Tesla T4 GPU)

### 1. Vector Addition: CPU vs. GPU Throughput Crossover
*Tiny workloads are dominated by ~5–15 µs kernel dispatch latency. Large workloads fully saturate the 2,560 cores:*

| Vector Size ($N$) | CPU Time (ms) | GPU Time (ms) | Speedup | Winner | Systems Reality |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **100** | 0.034 | 0.055 | 0.6x | CPU | Cache is warm, CPU finishes before GPU launch completes |
| **10,000** | 0.028 | 0.039 | 0.7x | CPU | Still bounded by kernel launch overhead |
| **100,000** | 0.198 | 0.109 | 1.8x | GPU | GPU cores begin to saturate |
| **1,000,000** | 2.037 | 0.077 | 26.5x | GPU | Massive thread parallelism wins |
| **10,000,000** | 22.169 | 0.713 | **31.1x** | GPU | 2,560 cores fully saturated 🚀 |

*(Note: The very first invocation on GPU includes a one-time ~30 ms CUDA context init & memory allocator pool setup).*

---

### 2. Matrix Multiplication (FP32): Naive vs. Shared Memory Tiled
*At $1024 \times 1024$, each thread in the Naive kernel re-reads the entire row and column from slow DRAM ($1,024 \times$ redundant reads). Shared memory tiling caches $16 \times 16$ blocks in ultra-fast L1 SRAM, slashing global memory traffic by up to $16\times$:*

| Matrix Size | cuBLAS (NVIDIA) | Naive CUDA | Tiled CUDA (SRAM) | Tiled Speedup | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **128 x 128** | 0.022 ms | 0.028 ms | 0.023 ms | **1.26x** | ✅ Bit-exact pass |
| **256 x 256** | 0.055 ms | 0.160 ms | 0.109 ms | **1.48x** | ✅ Bit-exact pass |
| **512 x 512** | 0.137 ms | 1.148 ms | 0.744 ms | **1.54x** | ✅ Bit-exact pass |
| **1024 x 1024** | 0.831 ms | 9.190 ms | 3.015 ms | **3.05x** 🚀 | ✅ Bit-exact pass |
| **2048 x 2048** | 4.257 ms | 46.041 ms | 30.236 ms | **1.52x** | ✅ Bit-exact pass |

---

### 3. Global Memory Coalescing & Shared Memory Bank Conflicts

#### Experiment 1: Global Memory Coalescing (256 MB read/write)
When consecutive threads in a warp access non-consecutive addresses, the hardware memory controller cannot coalesce them into a single 128-byte DRAM burst:

| Access Pattern | Bandwidth (GB/s) | Relative Efficiency | Memory Controller Impact |
| :--- | :--- | :--- | :--- |
| **Stride 1 (Coalesced)** | **221.74 GB/s** | **100.0%** | Single 128-byte burst transaction per warp |
| **Stride 2** | 163.56 GB/s | 73.8% | 26.2% bandwidth lost |
| **Stride 4** | 101.18 GB/s | 45.6% | 54.4% bandwidth lost |
| **Stride 8** | 55.69 GB/s | 25.1% | 74.9% bandwidth lost |
| **Stride 16** | 28.04 GB/s | 12.6% | 87.4% bandwidth lost |
| **Stride 32 (Worst Case)** | **25.53 GB/s** | **11.5%** | **88.5% Bandwidth Collapsed** (32 separate memory transactions!) |

#### Experiment 2: Shared Memory Bank Conflicts (100,000 accesses/thread)
Shared memory is divided into 32 banks (4 bytes wide). If multiple threads in a warp access different addresses within the same bank, accesses are serialized:

| Access Pattern | Kernel Time (ms) | Slowdown Factor | Silicon Explanation |
| :--- | :--- | :--- | :--- |
| **Stride 1 (0 Conflicts)** | **5.636 ms** | **1.00x Baseline** | All 32 threads hit unique banks (Banks 0..31 simultaneously) |
| **Stride 2 (2-Way Conflict)** | 8.356 ms | 1.48x slower | 2 threads hit the same bank (2 serialized phases) |
| **Stride 4 (4-Way Conflict)** | 14.679 ms | 2.60x slower | 4 threads hit the same bank (4 serialized phases) |
| **Stride 8 (8-Way Conflict)** | 28.001 ms | 4.97x slower | 8 threads hit the same bank (8 serialized phases) |
| **Stride 16 (16-Way Conflict)** | 56.077 ms | 9.95x slower | 16 threads hit the same bank (16 serialized phases) |
| **Stride 32 (32-Way Conflict)** | **112.239 ms** | **19.91x slower** | **ALL 32 threads serialized on Bank 0!** |

---

## 🚀 How to Run

### Option A: Interactive Colab Notebook (Recommended)
1. Open [Google Colab](https://colab.research.google.com/) and connect to a free **T4 GPU** runtime (`Runtime` $\to$ `Change runtime type` $\to$ `T4 GPU`).
2. Upload and open [`CUDA_Masterclass.ipynb`](CUDA_Masterclass.ipynb).
3. Execute the cells sequentially to compile kernels via PyTorch's `load_inline` JIT `nvcc` pipeline and run the live microbenchmarks.

### Option B: Standalone Python Script Execution
Clone this repository directly inside Colab or any CUDA-enabled Linux environment:
```bash
git clone https://github.com/puneethr09/ai_ms_python.git
cd ai_ms_python/learn_projects/cuda_kernels
python colab_all_in_one.py
```

# LESSON 3: Memory Hierarchy, Allocators, and Real Hardware Benchmarks

---

## 3.1 The Asynchronous GPU Execution & Hardware Event Model

In GPU programming, all kernel launches are **asynchronous** from the CPU's perspective:
1. Calling a kernel in Python/C++ does **not** execute immediately on the GPU ALUs.
2. The CPU simply pushes a command packet into the GPU's hardware FIFO queue (**CUDA Stream**).
3. The CPU moves to the next line of code in nanoseconds without waiting.

### Why `torch.cuda.synchronize()` is Mandatory for CPU Timing
- `time.perf_counter()` on the CPU only measures the **0.001 ms CPU dispatch time**.
- `torch.cuda.synchronize()` blocks the CPU host thread until the GPU hardware signals that all active warps across all SMs have completed execution.

### How `torch.cuda.Event` Works (GPU Hardware Timestamps)
```
THE GPU HARDWARE FIFO QUEUE (Stream)
───────────────────────────────────
CPU pushes into queue ──► [ Task 1: Record START timestamp ]
                          [ Task 2: Execute KERNEL (a + b) ]
                          [ Task 3: Record END timestamp   ]
                                       │
                                       ▼
                       GPU Silicon executes them sequentially:
                       1. Reaches Task 1 ──► Saves GPU Clock Cycle: T_start
                       2. Reaches Task 2 ──► 2,560 cores run computation
                       3. Reaches Task 3 ──► Saves GPU Clock Cycle: T_end
```
- `start.record()` and `end.record()` are executed **by the GPU hardware itself** in the command stream.
- `start.elapsed_time(end)` calculates exact elapsed time on the GPU silicon, completely eliminating CPU latency, OS thread scheduling, and PCIe queue dispatch delay.

---

## 3.2 Memory Allocation Mechanics: CPU vs GPU

### CPU Virtual Memory & `malloc()`
- **MMU Hardware Constraint:** The CPU hardware MMU cannot allocate arbitrary byte counts. It maps physical RAM in **4 KB Pages (4,096 bytes)** via page tables.
- **Syscall Overhead:** Asking the OS kernel for memory via `sys_brk()` or `sys_mmap()` requires switching CPU privilege from Ring 3 (User Mode) to Ring 0 (Kernel Mode), pausing the CPU pipeline, updating page tables, and flushing TLB translation caches (~1,000-2,000 CPU cycles / 1-2 microseconds).
- **User-Space Arena Pooling:** `glibc malloc` asks the OS kernel for a large chunk (**128 KB - 2 MB**) on first touch, then satisfies subsequent small allocations from its user-space arena in ~10 nanoseconds with zero syscalls.

### GPU `cudaMalloc()` vs PyTorch Caching Allocator
- **The Latency Gap:** `cudaMalloc` takes **0.5 to 5 milliseconds** because the CPU must send a request over the PCIe bus, the NVIDIA driver must synchronize the device, the GPU Memory Management Unit (GMMU) updates hardware page tables, and the result is signaled back over PCIe. This is roughly 1,000x slower than a CPU `mmap()` syscall.
- **PyTorch Caching Allocator:** On the first tensor allocation (**cold-start**), PyTorch calls `cudaMalloc` to claim a large VRAM memory pool (typically hundreds of MB to several GB, depending on available VRAM).
- All subsequent tensor allocations are sub-allocated from this pool in user-space in ~0.0001 ms (just pointer arithmetic, zero driver calls), enabling fast inference and training.

---

## 3.3 Live Tesla T4 Hardware Benchmark Analysis

### 1. Milestone 0: CPU vs GPU Throughput Crossover (Vector Addition)
| Vector Size (N) | CPU Time (ms) | GPU Time (ms) | Speedup | Winner | Physical Explanation |
|---|---|---|---|---|---|
| 100 | 8.201 | 34.437 | 0.2x | CPU | Cold-start: CUDA context init + PyTorch allocator pool creation (~30 ms one-time cost) |
| 1,000 | 0.034 | 0.055 | 0.6x | CPU | Cache is warm, CPU finishes before GPU dispatch completes |
| 10,000 | 0.028 | 0.039 | 0.7x | CPU | Still bounded by kernel launch overhead (~5-15 microseconds) |
| 100,000 | 0.198 | 0.109 | 1.8x | GPU | GPU cores begin to saturate |
| 1,000,000 | 2.037 | 0.077 | 26.5x | GPU | Massive parallelism wins |
| 10,000,000 | 22.169 | 0.713 | 31.1x | GPU | 2,560 cores fully saturated (31x speedup) |

---

### 2. Milestone 4.1 vs 4.2: Matrix Multiply (Tesla T4)
| Matrix Size | cuBLAS (ms) | Naive CUDA (ms) | Tiled CUDA (ms) | Tiled Speedup | Status |
|---|---|---|---|---|---|
| 128x128 | 0.022 | 0.028 | 0.023 | 1.26x | ✅ Bit-exact pass |
| 256x256 | 0.055 | 0.160 | 0.109 | 1.48x | ✅ Bit-exact pass |
| 512x512 | 0.137 | 1.148 | 0.744 | 1.54x | ✅ Bit-exact pass |
| 1024x1024 | 0.831 | 9.190 | 3.015 | **3.05x** | ✅ Bit-exact pass |
| 2048x2048 | 4.257 | 46.041 | 30.236 | 1.52x | ✅ Bit-exact pass |

#### Why Naive is 11x slower than cuBLAS at 1024x1024:
Every thread in Naive reads an entire row of A and an entire column of B from off-chip DRAM on every single loop iteration. For a 1024x1024 matrix, every single float is fetched **1,024 times from slow VRAM** (400 clock cycles per fetch).

#### Why Shared Memory Tiling is 3.05x faster:
The 256 threads in a 16x16 block collaborate to load a 16x16 tile into on-chip L1 SRAM **once**. All 16 threads in that row reuse the cached SRAM data for 16 multiply-accumulate operations at **1-2 clock cycles latency** instead of 400 cycles, slashing global memory traffic by up to 16x!

---

### 3. Memory Coalescing & Bank Conflict Microbenchmarks
```
EXPERIMENT 1: GLOBAL MEMORY COALESCING (256 MB read/write)
Stride 1  (Coalesced)  : 221.74 GB/s (100% Efficiency - 128-byte burst transaction)
Stride 2  (Strided)    : 163.56 GB/s (26.2% Bandwidth Lost)
Stride 4  (Strided)    : 101.18 GB/s (54.4% Bandwidth Lost)
Stride 8  (Strided)    :  55.69 GB/s (74.9% Bandwidth Lost)
Stride 16 (Strided)    :  28.04 GB/s (87.4% Bandwidth Lost)
Stride 32 (Worst Case) :  25.53 GB/s (88.5% Bandwidth Lost - 32 separate transactions!)

EXPERIMENT 2: SHARED MEMORY BANK CONFLICTS (100,000 accesses/thread)
Stride 1  (0 Conflicts):   5.636 ms (1.00x Baseline - all 32 threads hit unique banks)
Stride 2  (2-Way)      :   8.356 ms (1.48x slower - 2 threads hit same bank)
Stride 4  (4-Way)      :  14.679 ms (2.60x slower - 4 threads hit same bank)
Stride 8  (8-Way)      :  28.001 ms (4.97x slower - 8 threads hit same bank)
Stride 16 (16-Way)     :  56.077 ms (9.95x slower - 16 threads hit same bank)
Stride 32 (32-Way)     : 112.239 ms (19.91x slower - ALL 32 threads serialized on Bank 0!)
```

Note on Experiment 3 (Padding Test): In our run, the padded tile showed no measurable speedup (0.97x).
This is because our microbenchmark kernel was compute-bound rather than memory-latency-bound in that
specific configuration. In production tiled matrix multiply kernels where column access patterns
dominate, the padding trick (tile[32][33] instead of tile[32][32]) provides meaningful speedups.

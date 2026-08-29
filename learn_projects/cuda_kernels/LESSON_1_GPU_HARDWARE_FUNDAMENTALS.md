# LESSON 1: GPU Hardware Fundamentals (Complete Reference)

## 1.1 CPU vs GPU Architecture Philosophy

### CPU: Low Latency, Single-Thread Perfection
- 8–16 cores, each running at 5.0–5.8 GHz
- 90% of transistors spent on: Branch Prediction, Out-of-Order Execution, Speculative Execution, massive L1/L2/L3 caches
- Goal: Get ONE answer to ONE thread as fast as possible (~60ns memory latency)
- Each core has its own independent Instruction Pointer (Program Counter)
- Threads are truly independent — can branch, jump, and execute completely different code paths

### GPU: High Throughput, Massive Parallelism
- 40–144 Streaming Multiprocessors (SMs), each with 64–128 CUDA cores
- 90% of transistors spent on: Arithmetic Logic Units (ALUs / CUDA Cores)
- Goal: Get 10,000+ answers simultaneously (higher latency ~300ns, but 10x–40x bandwidth)
- Runs at lower clock speed (1.4–2.1 GHz) to prevent 10,000 cores from melting the chip
- Threads are grouped into Warps of 32 that share ONE instruction dispatcher

---

## 1.2 The Warp: 32 Threads in Lockstep (SIMT)

- All 32 threads in a Warp share a single instruction unit
- They execute the EXACT SAME instruction in each clock cycle (SIMD-like)
- This is called SIMT: Single Instruction, Multiple Threads

### Branch Divergence (The if/else Trap)
- On CPU: Each core has its own Instruction Pointer → threads branch independently
- On GPU: All 32 threads in a warp must execute the same instruction
  - If some threads take `if` and some take `else`:
    1. Hardware executes the `if` path with even threads active, odd threads MASKED OFF (idle)
    2. Then executes the `else` path with odd threads active, even threads MASKED OFF
    3. Total time = if_time + else_time (50% efficiency!)
  - If ALL 32 threads take the same branch → 100% efficiency (zero divergence)
  - Threads in DIFFERENT warps can diverge freely — no penalty

---

## 1.3 Latency Hiding (The GPU's Killer Feature)

When a warp requests data from VRAM (takes ~400 clock cycles):
1. The GPU does NOT stall or throw the warp away
2. The Warp Scheduler instantly swaps to another ready warp in 0 clock cycles (zero-cost context switch!)
3. All warp variables stay frozen in dedicated hardware registers (no save/restore needed)
4. By the time the scheduler cycles back, the VRAM data has arrived
5. The warp wakes up right where it left off and finishes its work

### Why this works without causing glitches:
- 400 clock cycles at 1.7 GHz = only 0.23 microseconds
- A 240 FPS display gives the GPU 4.1 milliseconds per frame
- 0.23 microseconds is only 0.005% of a single frame budget
- The GPU uses Double Buffering: renders into a hidden back buffer, then swaps the entire completed frame onto the display all at once

### Golden Rule of GPU Occupancy:
Launch WAY more threads than physical cores (50,000 to 1,000,000+) so the warp scheduler always has active warps ready to execute while others wait for memory.

---

## 1.4 The GPU Memory Hierarchy (Tesla T4 / RTX 3090)

### Speed & Latency Ladder:

| Tier | Physical Location | Size | Latency | Bandwidth |
|---|---|---|---|---|
| Registers | On the ALU inputs | 256 KB/SM | 0-1 cycles | ~60 TB/s aggregate (theoretical peak) |
| L1/Shared Memory | Inside each SM (on-chip SRAM) | 64 KB/SM (T4) or 128 KB/SM (3090) | 1-4 cycles (conflict-free) | ~19 TB/s aggregate |
| L2 Cache | Center of GPU die | 4 MB (T4) / 6 MB (3090) shared | 200-300 cycles | ~3.5 TB/s |
| VRAM (Global Memory) | Off-chip DRAM chips | 15 GB (T4) / 24 GB (3090) | 400-600 cycles | 0.3-0.9 TB/s |

### Key Insight:
VRAM is 400x slower (latency) and 20x less bandwidth than L1 Shared Memory.
The entire art of CUDA optimization = move data from Global → Shared → Registers, compute, write back.

### NVIDIA's Unique Trick: Configurable L1/Shared Memory
The on-chip SRAM per SM is programmer-configurable:
- You can allocate part as automatic L1 cache (hardware controlled)
- And part as `__shared__` memory (YOU control it explicitly in code)
- Writing `__shared__ float s_A[16][16]` reserves a slice of ultra-fast on-chip SRAM

---

## 1.5 Physical Wiring: VRAM → L2 → L1 → ALUs

VRAM does NOT connect directly to SMs. The path is:

1. VRAM (off-chip GDDR6/HBM chips) → Memory Controllers via physical bus (384-bit on RTX 3090)
2. Memory Controllers → L2 Cache (on-chip, shared across all SMs)
3. L2 Cache → SMs via High-Speed Crossbar Network-on-Chip (NoC)
4. Inside each SM: L1/Shared Memory → Register File → ALU inputs

### Why L1 is called "Data" Cache:
- Each SM has a tiny L1 Instruction Cache (~32KB) but GPU kernels are small (~150 instructions), so it never misses
- AI data tensors are massive (billions of floats, 16–24 GB), so 99.999% of memory traffic is data traffic

---

## 1.6 CPU vs GPU Bus Width Comparison

| Tier | CPU (Desktop) | GPU (RTX 3090) |
|---|---|---|
| Off-chip RAM bus | 128-bit (Dual Channel DDR5) | 384-bit (GDDR6X) |
| RAM Bandwidth | ~90 GB/s | 936 GB/s |
| L2 Interconnect | 512-bit Ring Bus (~600 GB/s) | Crossbar NoC (~3,500 GB/s) |
| L1 Aggregate BW | ~3,000 GB/s | ~19,000 GB/s |
| Register Aggregate BW | ~6,000 GB/s | ~60,000 GB/s |

---

## 1.7 Silicon Interposer & HBM (High Bandwidth Memory)

### Traditional PCB (RTX 3090):
- GPU chip in center, GDDR6 chips soldered around it on fiberglass PCB
- Copper traces on PCB limited to ~0.1mm spacing → max ~384 wires
- Bandwidth: ~936 GB/s

### 2.5D Packaging with Silicon Interposer (H100, B200):
- HBM3 Stacks: 8–16 DRAM dies stacked vertically like pancakes, connected by TSVs (Through-Silicon Vias)
- Silicon Interposer: Wafer-thin silicon slab underneath GPU and HBM stacks
- TSMC lithography etches 5,000–10,000+ microscopic wires connecting GPU to HBM stacks micrometers away
- Result: 5,120+ bit bus width, 3,350 GB/s (H100) to 8,000 GB/s (B200)

---

## 1.8 Why GPU Memory Latency > CPU Memory Latency

Four physical reasons:
1. **Lower Clock Speed:** GPU cores at 1.7 GHz (0.5 ns/cycle) vs CPU at 5.5 GHz (0.17 ns/cycle)
2. **Massive Traffic Arbitration:** Memory controllers serving 100,000+ thread requests vs CPU serving 8–16 cores
3. **DRAM Topology:** GDDR6/HBM optimized for wide burst throughput, not low-latency random access
4. **Intentional Design Trade-off:** GPU architects accepted higher latency because warp schedulers hide it; in exchange they gained 10x–40x total bandwidth

---

## 1.9 Compute Units Inside One SM

| Unit | Tesla T4 (Turing) | RTX 3090 (Ampere) | What It Does |
|---|---|---|---|
| FP32 CUDA Cores | 64 per SM | 128 per SM | Scalar multiply-add: D = A*B + C (2 FLOPs/cycle) |
| Tensor Cores | 8 per SM (2nd Gen) | 4 per SM (3rd Gen, more powerful per core) | Matrix multiply-add on 4x4 or 8x8 sub-tiles |
| SFUs (Special Function Units) | 4 per SM | 4 per SM | Fast sin, cos, exp, rsqrt via hardware lookup (~8 cycles, pipelined) |
| Warp Schedulers | 4 per SM | 4 per SM | Issue 4 warps (128 threads) per clock cycle |
| Register File | 256 KB per SM | 256 KB per SM | 65,536 32-bit registers divided among all active threads |

### Tensor Cores: The AI Supercharger
- Even though Ampere has fewer Tensor Cores per SM (4 vs 8), each 3rd-gen core is significantly more powerful than a 2nd-gen core
- Tensor Cores perform small matrix multiply-accumulate operations (e.g. 4x4x4) in a single cycle
- Combined throughput per SM is hundreds of FP16 TFLOPS, far exceeding what CUDA cores can achieve
- ~98% of LLM computation is matrix multiply (handled by Tensor Cores)
- ~2% is non-linear operations like softmax, RMSNorm (handled by CUDA cores and SFUs)

---

## 1.10 Tesla T4 vs RTX 3090 Spec Sheet

| Component | Google Colab (Tesla T4) | Office (RTX 3090) |
|---|---|---|
| SM Count | 40 | 82 |
| Total CUDA Cores | 2,560 (40 x 64) | 10,496 (82 x 128) |
| VRAM | 15 GB GDDR6 (15,360 MiB) | 24 GB GDDR6X |
| Memory Bandwidth | 320 GB/s | 936 GB/s |
| FP32 Peak | 8.1 TFLOPS | 35.6 TFLOPS |
| Tensor Cores | 320 total (8 per SM, Turing 2nd Gen) | 328 total (4 per SM, Ampere 3rd Gen) |

---

## 1.11 Benchmark Results (Verified on Tesla T4 via CUDA_Masterclass.ipynb)

### Milestone 0: CPU vs GPU Vector Operations
| Vector Size | CPU (ms) | GPU (ms) | Speedup | Winner |
|---|---|---|---|---|
| 100 | 8.201 | 34.437 | 0.2x | CPU (cold-start overhead dominates) |
| 1,000 | 0.034 | 0.055 | 0.6x | CPU |
| 10,000 | 0.028 | 0.039 | 0.7x | CPU |
| 100,000 | 0.198 | 0.109 | 1.8x | GPU |
| 1,000,000 | 2.037 | 0.077 | 26.5x | GPU |
| 10,000,000 | 22.169 | 0.713 | 31.1x | GPU |

Note: Row 1 (N=100) includes CUDA context initialization and PyTorch caching allocator cold-start (~30 ms one-time cost).
Key insight: Below N=10,000, kernel launch overhead (~5-15 us) dominates. Above N=100,000, GPU massively wins.

### Milestone 4.1 & 4.2: Naive vs Tiled CUDA Matrix Multiply
| Matrix Size | NVIDIA cuBLAS | Naive CUDA | Tiled CUDA | Tiled Speedup |
|---|---|---|---|---|
| 128x128 | 0.022 ms | 0.028 ms | 0.023 ms | 1.26x |
| 256x256 | 0.055 ms | 0.160 ms | 0.109 ms | 1.48x |
| 512x512 | 0.137 ms | 1.148 ms | 0.744 ms | 1.54x |
| 1024x1024 | 0.831 ms | 9.190 ms | 3.015 ms | 3.05x |
| 2048x2048 | 4.257 ms | 46.041 ms | 30.236 ms | 1.52x |

Key insight: Shared memory tiling reduced global memory traffic, achieving 3.05x speedup at 1024x1024.
Naive kernel is 11x slower than cuBLAS because every float is re-read 1,024 times from VRAM.

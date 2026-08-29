# 🚀 AI Systems Engineering, C++/Python Inference & CUDA Masterclass
> **From Bare-Metal Silicon & Linux Virtual Memory to Custom CUDA Kernels, llama.cpp, and Distributed LLM Inference**

---

## 📖 About This Repository
This repository is a comprehensive, production-grade learning ledger and systems portfolio for high-performance AI systems engineering. It maps every layer of the compute stack—from CPU silicon logic gates and Linux OS virtual memory to custom CUDA C++ kernels and enterprise-scale LLM inference engines (`llama.cpp`, `vLLM`, TensorRT-LLM).

---

## 📚 The Modular Knowledge Base (Living Textbook)

The theoretical foundations and physical principles are documented across dedicated, publication-grade chapters:

| Level | Chapter & Topic | Core Systems Concepts |
| :--- | :--- | :--- |
| **01** | ⚡ [**Bare-Metal Silicon & The Latency Pyramid**](file:///Users/puneeth/repo/ai_ms_python/knowledge/01_hardware_silicon.md) | Flip-flop registers, 6T-SRAM vs 1T-1C DRAM, $RC$ wire delay, Set-Associativity, LRU cache circuits |
| **02** | 🖥️ [**OS, Virtual Memory & RAII**](file:///Users/puneeth/repo/ai_ms_python/knowledge/02_os_virtual_memory.md) | 64-bit address space, Demand Paging, BSS/Data/rodata segments, `mmap()`, RAII stack unwinding |
| **03** | 🛡️ [**Processes, Threads & Systems Security**](file:///Users/puneeth/repo/ai_ms_python/knowledge/03_processes_threads_security.md) | Page table isolation, Chrome/VS Code/Games architectures, DLL injection, buffer overflows, Spectre |
| **04** | 🔄 [**Concurrency & Asynchronous Engines**](file:///Users/puneeth/repo/ai_ms_python/knowledge/04_concurrency_async_engines.md) | C++20 `std::jthread`, Python GIL release mechanics, `epoll`/`kqueue` event loops, C10k web servers |
| **05** | 📦 [**Python Internals & Memory Models**](file:///Users/puneeth/repo/ai_ms_python/knowledge/05_python_internals_memory.md) | `PyObject` C-struct layout, Refcounting vs Cyclic GC, `pymalloc` arenas/pools, Zero-Copy `Py_buffer` |
| **06** | 🚀 [**Inference Engineering Roadmap**](file:///Users/puneeth/repo/ai_ms_python/knowledge/06_inference_engineering_roadmap.md) | Memory bandwidth physics, Inverse Law of token speed, `mlock` page eviction, `llama.h` C++ API |
| **07** | ⚡ [**CUDA Lesson 1: GPU Hardware Fundamentals**](file:///Users/puneeth/repo/ai_ms_python/learn_projects/cuda_kernels/LESSON_1_GPU_HARDWARE_FUNDAMENTALS.md) | SMs, 2,560 cores, SIMT Warps, Zero-Cost Context Switching, Latency Hiding, Tensor Cores |
| **08** | 🧩 [**CUDA Lesson 2: Programming & Coordinate Model**](file:///Users/puneeth/repo/ai_ms_python/learn_projects/cuda_kernels/LESSON_2_CUDA_PROGRAMMING_MODEL.md) | Grid/Block/Thread 1D & 2D coordinate formulas, GigaThread block scheduler, Boundary guards |
| **09** | 🔬 [**CUDA Lesson 3: Memory Hierarchy & Benchmarks**](file:///Users/puneeth/repo/ai_ms_python/learn_projects/cuda_kernels/LESSON_3_MEMORY_HIERARCHY_AND_BENCHMARKS.md) | Asynchronous CUDA Streams, `cudaMalloc` vs Caching Allocator, 32-way bank conflicts, live T4 benchmarks |

---

## 🛠️ Hands-On Applied Projects

Each project implements clean, benchmarked, production-ready code:

### 1. ⚡ [Custom CUDA Matrix Multiplication Kernels](file:///Users/puneeth/repo/ai_ms_python/learn_projects/cuda_kernels/)
- 📓 [`CUDA_Masterclass.ipynb`](file:///Users/puneeth/repo/ai_ms_python/learn_projects/cuda_kernels/CUDA_Masterclass.ipynb): Interactive Colab/Tesla T4 GPU notebook.
- 📄 [`00_gpu_fundamentals.py`](file:///Users/puneeth/repo/ai_ms_python/learn_projects/cuda_kernels/00_gpu_fundamentals.py): GPU hardware inspection and CPU vs GPU crossover benchmark.
- 📄 [`01_naive_matmul.py`](file:///Users/puneeth/repo/ai_ms_python/learn_projects/cuda_kernels/01_naive_matmul.py): First raw C++ CUDA kernel (JIT-compiled with `nvcc` via PyTorch `load_inline`).
- 📄 [`02_shared_memory_tiled.py`](file:///Users/puneeth/repo/ai_ms_python/learn_projects/cuda_kernels/02_shared_memory_tiled.py): Shared memory tiled matrix multiplication (**3.05x speedup** on Tesla T4).
- 📄 [`03_coalescing_and_bank_conflicts.py`](file:///Users/puneeth/repo/ai_ms_python/learn_projects/cuda_kernels/03_coalescing_and_bank_conflicts.py): Microbenchmarks proving **88.5% bandwidth loss** on strided access and **19.9x slowdown** on 32-way shared memory bank conflicts.

### 2. ⚡ [C++ / Python Hybrid Zero-Copy API](file:///Users/puneeth/repo/ai_ms_python/learn_projects/hybrid_api/)
- High-performance FastAPI server backed by a compiled C++ engine (`pybind11`).
- Demonstrates zero-copy buffer sharing (`Py_buffer`) and GIL release (`py::gil_scoped_release`) for non-blocking concurrent request handling.

### 3. 🦙 [Custom llama.cpp C++ Inference Engine](file:///Users/puneeth/repo/ai_ms_python/learn_projects/custom_inference/)
- Bare-metal C++ inference driver executing quantized LLMs (Qwen-2.5 7B, Llama-3.1 8B) on Apple Silicon Metal GPU / Unified Memory.

### 4. 🗂️ Systems & Data Utilities
- 📂 [`learn_projects/smart_file_organizer/`](file:///Users/puneeth/repo/ai_ms_python/learn_projects/smart_file_organizer/): High-throughput CLI file organizer with generators and atomic POSIX operations.
- 📂 [`learn_projects/hackernews_scraper/`](file:///Users/puneeth/repo/ai_ms_python/learn_projects/hackernews_scraper/): Concurrent API scraper and data processing pipeline.

---

## 📊 Silicon Benchmark Summary (Tesla T4 GPU)

Verified empirical results from our live JIT kernel execution sessions:

```
1. CPU vs GPU Throughput Crossover (Vector Add):
   - N = 100       : CPU wins (0.034 ms vs GPU 0.055 ms) due to ~5-15 µs kernel launch overhead.
   - N = 1,000,000 : GPU wins (0.077 ms vs CPU 2.037 ms) -> 26.5x Speedup 🚀
   - N = 10,000,000: GPU wins (0.713 ms vs CPU 22.169 ms) -> 31.1x Speedup 🚀

2. Matrix Multiplication at 1024x1024 (FP32):
   - Naive CUDA Kernel  : 9.190 ms (11x slower than cuBLAS due to 1,024 redundant DRAM reads per thread)
   - Tiled CUDA (SRAM)  : 3.015 ms (3.05x faster than Naive by caching 16x16 tiles in L1 SRAM)
   - NVIDIA cuBLAS Peak : 0.831 ms

3. Memory Coalescing & Bank Conflicts:
   - Coalesced Global Reads (Stride 1) : 221.74 GB/s (100% bus efficiency)
   - Non-Coalesced Reads   (Stride 32):  25.53 GB/s (88.5% bandwidth collapsed!)
   - Conflict-Free Shared Mem Access  :   5.636 ms
   - 32-Way Serialized Bank Conflict   : 112.239 ms (19.9x slowdown!)
```

---

## 🗺️ Operational Blueprints
- 📋 [**Edge Lab & Operations Blueprint**](file:///Users/puneeth/repo/ai_ms_python/EDGE_LAB_BLUEPRINT.md): Architecture blueprint for Raspberry Pi 5 cluster, Pironman 5 NVMe setup, AI Overwatcher telemetry, and Tailscale mesh networking.

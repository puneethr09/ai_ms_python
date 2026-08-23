# Stage 4: GPU Programming & Custom CUDA Kernels

## 🧠 The Mental Model Shift (CPU → GPU)

You've spent your career writing code that runs on **1-8 fast cores** (CPU).
A GPU has **thousands of slow cores** that all execute the **same instruction** simultaneously.

This is not a minor optimization trick — it is a fundamentally different way of thinking about computation.

---

## 📁 Project Structure

```
cuda_kernels/
├── README.md                          ← You are here
├── 00_gpu_fundamentals.py             ← GPU vs CPU mental model (runs on Colab)
├── 01_naive_matmul.py                 ← Milestone 4.1: First CUDA kernel
├── 02_shared_memory_tiled.py          ← Milestone 4.2: Shared memory optimization
├── 03_register_tiled.py               ← Milestone 4.3: Register tiling + float4
├── 04_pytorch_extension/              ← Milestone 4.4: Custom PyTorch op
└── 05_benchmark_vs_cublas.py          ← Milestone 4.5: Performance comparison
```

## 🚀 How to Run

All `.py` files are designed to run directly in **Google Colab** (free T4 GPU).

1. Open [Google Colab](https://colab.research.google.com/)
2. Runtime → Change runtime type → **T4 GPU**
3. Upload or paste the `.py` file contents into a cell
4. Run

## 🎯 Learning Goals

| Milestone | What You Learn | C++ Analogy |
|---|---|---|
| 0 | Why GPUs exist, thread hierarchy, memory hierarchy | `std::thread` but 10,000 of them |
| 1 | Write `__global__` kernel, launch grid, global memory | Writing a function that 65,536 threads run simultaneously |
| 2 | `__shared__` memory tiling, `__syncthreads()` | Like a per-block L1 cache you control manually |
| 3 | Register pressure, `float4` vectorized loads, ILP | Like SIMD intrinsics (`_mm256_load_ps`) but for GPU |
| 4 | PyTorch C++ extension with pybind11 | Your Stage 2 C++ skills + CUDA = custom PyTorch ops |
| 5 | Profiling, roofline model, GFLOPS | Like `perf stat` but for GPU (`nvprof`, `nsight`) |

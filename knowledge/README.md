# 🧠 The Systems Engineering & Inference Masterclass
> **From Bare-Metal Silicon to Enterprise AI: Modern C++ (17/20), Python Internals, and High-Performance Inference**

---

## 📑 Modular Master Index

The complete knowledge base is modularized into dedicated, deep-dive chapters:

| Chapter | Topic & Focus Area | Key Systems Concepts Covered |
| :--- | :--- | :--- |
| ⚡ [**01. Bare-Metal Silicon**](file:///Users/puneeth/repo/ai_ms_python/knowledge/01_hardware_silicon.md) | Hardware Silicon & Latency | Registers, Latency Pyramid, 6T-SRAM vs 1T-1C DRAM, $O(1)$ Circuit Physics, LRU |
| 🖥️ [**02. OS & Virtual Memory**](file:///Users/puneeth/repo/ai_ms_python/knowledge/02_os_virtual_memory.md) | Virtual Addressing & Memory Layout | 64-bit Address Map, MMU/Paging, BSS vs Heap, `mmap`, RAII & Smart Pointers |
| 🛡️ [**03. Processes, Threads & Security**](file:///Users/puneeth/repo/ai_ms_python/knowledge/03_processes_threads_security.md) | Systems Architecture & Security | Chrome/Spotify/Games isolation, DLL Injection, Buffer Overflows, Spectre |
| 🔄 [**04. Concurrency & Async Engines**](file:///Users/puneeth/repo/ai_ms_python/knowledge/04_concurrency_async_engines.md) | Concurrency Models & Event Loops | Chef Analogy, C++20 `std::jthread`, Python GIL, Apache vs NGINX C10k |
| 📦 [**05. Python Internals & Memory Models**](file:///Users/puneeth/repo/ai_ms_python/knowledge/05_python_internals_memory.md) | Python C-API & Buffer Protocols | `PyObject` anatomy, Refcounting vs Cyclic GC, `pymalloc`, Zero-Copy Buffers, `pybind11` |
| 🚀 [**06. Inference Engineering Roadmap**](file:///Users/puneeth/repo/ai_ms_python/knowledge/06_inference_engineering_roadmap.md) | 5-Stage Hardware Blueprint | Mac $\to$ Raspi 5 $\to$ Colab CUDA $\to$ RTX 3090 Rig, Bandwidth Physics, PagedAttention |

---

## 🗺️ Visual Architecture Roadmap

```
Level 1: ⚡ Bare-Metal Silicon ──> [ 01_hardware_silicon.md ]
   │
Level 2: 🖥️ OS & Virtual Memory ──> [ 02_os_virtual_memory.md ]
   │
Level 3: 🛡️ Processes & Security ──> [ 03_processes_threads_security.md ]
   │
Level 4: 🔄 Concurrency & Async ──> [ 04_concurrency_async_engines.md ]
   │
Level 5: 📦 Language Memory ────> [ 05_python_internals_memory.md ]
   │
Level 6: 🚀 5-Stage Inference Roadmap ──> [ 06_inference_engineering_roadmap.md ]
         ├── Stage 1: Local Mac ────────> C++/Python Bridge (FastAPI + pybind11 Zero-Copy)
         ├── Stage 2: Local Mac (Metal) ─> llama.cpp Deep Dive (Llama-3.1 8B, KV-Cache in C++)
         ├── Stage 3: Raspberry Pi 5 ───> Edge CPU Optimization (ARM NEON SIMD, Extreme Constraints)
         ├── Stage 4: Google Colab ─────> Raw CUDA C++ Kernels (.cu, Thread Hierarchy, Shared Mem)
         └── Stage 5: Office RTX 3090 ──> Enterprise Scale (vLLM, TensorRT-LLM, FlashAttention-2)
```

---

## 📂 Applied Workspace Projects

* 📂 [`learn_projects/smart_file_organizer/`](file:///Users/puneeth/repo/ai_ms_python/learn_projects/smart_file_organizer/): Project 1 CLI Organizer (`pathlib`, generators, POSIX rename).
* 📂 [`learn_projects/hackernews_scraper/`](file:///Users/puneeth/repo/ai_ms_python/learn_projects/hackernews_scraper/): Project 2 API Cruncher (`requests`, safe map reads, list comprehensions).
* 📂 [`learn_projects/hybrid_api/`](file:///Users/puneeth/repo/ai_ms_python/learn_projects/hybrid_api/): Project 3 FastAPI + C++ `pybind11` Zero-Copy Server (GIL release, contiguous buffers).

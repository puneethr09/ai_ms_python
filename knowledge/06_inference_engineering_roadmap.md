# Level 6: The 5-Stage Inference Engineering Roadmap
> **Hardware-Mapped Strategy: From Local Silicon to Enterprise GPU Workstations**

---

## 1. The 5-Stage Hardware Blueprint

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              THE 5-STAGE INFERENCE JOURNEY                                  │
│                                                                                             │
│  [ Stage 1: Local Mac ] ──> [ Stage 2: Local Mac ] ──> [ Stage 3: Raspi 5 ]                │
│  The C++/Python Bridge       llama.cpp Deep Dive       Edge CPU Optimization                │
│                                                                                             │
│                             ──> [ Stage 4: Google Colab ] ──> [ Stage 5: RTX 3090 Rig ]     │
│                                  Raw CUDA C++ Kernels          Enterprise vLLM / 70B Model   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Stage | Focus Area | Device Used | Hardware Constraints & Goals |
| :--- | :--- | :--- | :--- |
| **Stage 1 (Now)** | C++/Python Hybrid Architecture | **Local Mac** | FastAPI + `pybind11` Zero-Copy Buffer Server |
| **Stage 2** | Production LLM C++ Internals | **Local Mac** | `llama.cpp` + Llama-3.1 8B (Metal GPU & Unified Memory) |
| **Stage 3** | Edge AI & Memory Constraints | **Raspberry Pi 5** | ARM NEON SIMD, CPU-Only Quantized 2B/3B Models |
| **Stage 4** | GPU Architecture & Raw CUDA | **Google Colab** | Custom CUDA C++ Matrix Multiply Kernels (`.cu`) |
| **Stage 5** | Enterprise Scale & Optimization | **Office RTX 3090** | vLLM + Llama-3-70B + PagedAttention + `nsys` Profiling |

---

## 2. Memory Bandwidth: The Core Inference Bottleneck

Think of Memory Bandwidth as the **width of the pipe** between RAM and compute cores:

```
[ Standard CPU RAM (DDR5) ] ════ (80 GB/sec) ════> [ CPU Cores ]

[ Apple Silicon Unified Memory ] ════════════════ (400 - 800 GB/sec) ════════> [ Mac GPU / CPU ]

[ NVIDIA H100 GPU (HBM3) ] ══════════════════════════════════════════════════> [ GPU Cores ]
                           (3,350 GB/sec — 35x wider than PC RAM!)
```

### Why Bandwidth Dictates LLM Token Speed:
* A **Llama 3 70B** model in 16-bit float is **140 GB**.
* Generating **1 single word** requires reading all **140 GB** of weights into compute cores once:
  * **On PC CPU (80 GB/s):** $140 / 80 = 1.75\text{ seconds per word}$ (Slow).
  * **On H100 GPU (3,350 GB/s):** $140 / 3,350 = 0.04\text{ seconds per word}$ (**25 words per second!**).

---

## 3. Why 16 GB RAM is Ideal for Mastering Quantization

| Model | Raw Size (16-bit Float) | 4-bit Quantized Size (GGUF) | Runs on Your 16 GB Mac? | Speed on Your Mac |
| :--- | :--- | :--- | :--- | :--- |
| **Llama 3.1 (8B)** | 16.0 GB (Tight) | **4.9 GB** | ✅ **Easily!** | **30 to 45 words/sec** |
| **Gemma 2 (2B)** | 4.5 GB | **1.6 GB** | ✅ **Instant!** | **60+ words/sec** |
| **Mistral (7B)** | 14.5 GB | **4.3 GB** | ✅ **Easily!** | **35 words/sec** |

---

## 4. OS Concepts Repurposed for AI Engines

* **Pages & Page Tables $\to$ PagedAttention (vLLM):** Divides KV-cache into discrete pages to eliminate 60–80% GPU memory fragmentation.
* **Free-Block Queues:** Manages available KV-cache blocks in GPU VRAM in $O(1)$ time.
* **Distributed Rank & World Size:** Splits 70B+ models across multi-GPU nodes using NCCL over NVLink (`world_size = total GPUs`, `rank = current GPU ID`).

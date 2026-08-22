# 🚀 Project 4: Custom Standalone C++ LLM Inference Engine

> **Bare-Metal Transformer Inference on Apple Metal GPU using the Native `llama.h` C++ API**

---

## 📌 Architecture Overview

This project demonstrates the complete, internal, unabstracted 7-step pipeline of a Large Language Model inference engine written in modern C++17.

```
[ User String Prompt ]
          │
          ▼
1. Tokenization (llama_tokenize) ─────────────> Converts text to integer token IDs via BPE.
          │
          ▼
2. Context Allocation (llama_init_from_model) ─> Allocates 14.00 MiB KV-Cache buffer in RAM.
          │
          ▼
3. Prefill Phase (llama_decode on prompt batch) -> Parallel Matrix Multiplication (GEMM) in 3.73 ms.
          │
          ▼
4. Decode Loop (llama_sampler_sample + decode) -> Autoregressive token-by-token generation (GEMV) at 17.05 t/s.
          │
          ▼
5. Teardown (llama_free + llama_model_free) ───> Releases KV-cache and calls munmap() on the 4.68 GB model.
```

---

## ⚙️ Compilation & Build Instructions

Compile with Apple Metal GPU backend support linked against `llama.cpp` shared libraries:

```bash
cd /Users/puneeth/repo/ai_ms_python/learn_projects/custom_inference

clang++ -std=c++17 -O3 \
  -I../llama.cpp/include -I../llama.cpp/ggml/include \
  -L../llama.cpp/build/bin \
  -lllama -lggml -lggml-metal -lggml-cpu -lggml-base \
  -Wl,-rpath,../llama.cpp/build/bin \
  custom_infer.cpp -o custom_infer
```

---

## 🏃 Execution

```bash
./custom_infer
```

### Measured Hardware Performance (Apple Silicon Mac):
* **Model Size:** 4.68 GB (Qwen 2.5 Coder 7B Q4_K_M)
* **Model Mapping (`mmap`):** **`624.09 ms`**
* **Prefill Time (TTFT):** **`3.73 ms`**
* **Generation Speed:** **`17.05 tokens/second`**
* **Memory Bandwidth Utilized:** **`~80 Gigabytes / second`**

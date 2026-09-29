# Level 6: The Inference Engineering Roadmap
> **Build every layer yourself, check it against a reference, and measure it against the hardware's limit.**
>
> 📗 The theory for each phase lives in [**The Inference Textbook**](inference_textbook/): how a transformer produces a word and why it's slow, with flashcards.

---

## 1. The 9-Phase Build Plan

```
┌──────────────────────────────── APPLE M3 (16 GB, ~100 GB/s) ─────────────────────────────────┐
│ [0] Instruments ─> [1] Transformer in numpy ─> [2] C engine (NEON) ─> [3] Quantization        │
│                                                     └─> [4] Metal / MLX ─> [6] Serving         │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
┌──── FREE COLAB T4 ────┐   ┌──── RENTED GPU (weekends) ────┐   ┌──── ONGOING ────────────────┐
│ [5] CUDA + Triton     │   │ [7] vLLM/SGLang, multi-GPU    │   │ [8] Upstream PRs, write-ups │
└───────────────────────┘   └───────────────────────────────┘   └─────────────────────────────┘
```

Hands-on code lives in [`learn_projects/inference_from_scratch/`](../learn_projects/inference_from_scratch/).

| Phase | Build | Hardware | Exit criterion (must be measured) |
| :--- | :--- | :--- | :--- |
| **0. Instruments** | Benchmark and test harness, STREAM-style bandwidth test for CPU and GPU | M3 | Your M3's real bandwidth (GB/s), recorded. Every later result is reported as a % of it |
| **1. Transformer from scratch** | BPE tokenizer, safetensors loader, Qwen2.5-0.5B forward pass (RMSNorm, RoPE, GQA, SwiGLU), sampling, all in numpy | M3 | Token IDs match HF exactly; logits match HF (fp32) within 1e-3, same argmax |
| **2. C engine** | Port the forward pass to C, `mmap` weights, KV cache, NEON matrix-vector kernel, threads | M3 CPU | Decode tokens/sec ≥ 70% of `bandwidth ÷ model bytes`; prefill and decode timed separately |
| **3. Quantization** | Q8_0 / Q4_0 block quantization, fused dequant + matrix-vector kernel, perplexity eval | M3 CPU | Speed vs perplexity table for FP16 / Q8 / Q4 on your own engine |
| **4. Apple GPU** | Metal compute kernels (matvec, RMSNorm, softmax, attention); same model in MLX | M3 GPU | Your Metal decode vs MLX vs llama.cpp Metal, as % of bandwidth |
| **5. CUDA** | Matvec kernel, register-tiled GEMM, fused softmax/RMSNorm, FlashAttention forward, then Triton versions | Colab T4 | GEMM ≥ 70% of cuBLAS; every kernel reported as % of peak |
| **6. Serving** | Continuous-batching scheduler, paged KV cache, prefix caching, streaming, speculative decoding, load tester | M3 | TTFT / TPOT / throughput curves under concurrent users |
| **7. Scale** | Read vLLM and SGLang, tensor parallel across 2 GPUs, FP8, Nsight Systems/Compute | Rented GPU | Profiled multi-GPU run with the bottleneck explained |
| **8. Show it** | Merged PRs to llama.cpp / MLX / vLLM, benchmark-driven write-ups | — | Public, reviewable work |

**When is a GPU needed?** Not before Phase 5, and Phase 5 runs on free Colab. Rent (about $0.5–3/hr) only for Phase 7. Buy only if Phase 7 becomes daily work.

**Rules:** every phase ends with a test against a reference and a benchmark against the hardware's limit. Code first, docs second.

### Reading list (read alongside the phases)
| Phase | Paper |
| :--- | :--- |
| 1 | Attention Is All You Need (Vaswani et al., 2017) · RoFormer / RoPE (Su et al., 2021) |
| 0, 2 | Roofline: An Insightful Visual Performance Model (Williams et al., 2009) |
| 3 | LLM.int8() · GPTQ · AWQ · SmoothQuant |
| 5 | FlashAttention 1 / 2 / 3 |
| 6 | Orca (continuous batching) · PagedAttention / vLLM · Speculative Decoding (Leviathan et al.) |
| 7 | DistServe (disaggregated prefill/decode) · Megatron-LM (tensor parallelism) |

---

## 2. Memory Bandwidth: The Core Inference Bottleneck

Memory Bandwidth is the **physical volume of data (in Gigabytes)** that can travel between RAM and the compute cores every single second.

### 📐 The Silicon Physics Formula:
$$\text{Theoretical Memory Bandwidth} = \frac{\text{Memory Bus Width (Bits)} \times \text{Clock Transfer Rate (MT/s)}}{8\text{ bits per byte}}$$

It is strictly determined by **two hardware factors**:
1. **Bus Width (The Width of the Highway):** The number of physical copper traces etched into the silicon connecting compute dies to DRAM (128-bit, 256-bit, 512-bit, 5120-bit).
2. **Clock Rate (The Speed of the Cars):** The transfer frequency of the memory chips (e.g. LPDDR5 @ 6,400 MT/s, GDDR6X @ 19.5 Gbps, HBM3 @ 3.2 GHz).

---

## 3. Capacity (RAM Size) vs. Bandwidth (Traffic Flow)

An LLM generating text does **not** keep numbers in registers permanently. To predict **one single token (word)**, the GPU must stream **every single weight in the entire neural network** across the memory bus:

```
┌────────────────────────────────────────────────────────┐
│  MODEL IN RAM (CAPACITY): Total Size = 4.68 GB         │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
To generate 1 word, the GPU must do 1 COMPLETE LAP over all 4.68 GB!
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ • Token 1:  GPU reads 4.68 GB from RAM                 │
│ • Token 2:  GPU reads 4.68 GB from RAM                 │
│ • ...                                                  │
│ • Token 17: GPU reads 4.68 GB from RAM                 │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
In 1 SECOND, the engine generates 17.05 tokens!
Data Traffic Streamed = 4.68 GB × 17.05 tokens/sec = 79.79 GB / sec!
```

---

## 4. The Inverse Law of Token Generation Speed

$$\text{Token Generation Speed (tokens/sec)} = \frac{\text{Sustained Memory Bandwidth (GB/s)}}{\text{Model Size in RAM (GB)}}$$

> [!NOTE]
> **Measured in Phase 0:** this M3 sustains **92 GB/s** (CPU) and **87 GB/s** (GPU) out of 102.4 GB/s theoretical ([`hardware.json`](../learn_projects/inference_from_scratch/results/hardware.json)). llama.cpp's 17.05 tok/s on the 4.68 GB model moves ~80 GB/s, about **92% of the measured GPU peak**. That is the bar your own engine has to reach.

Look at how token speed scales across model sizes on an **$80\text{ GB/s}$ Memory Bus (Your Mac)**:

| Model Size in RAM | Real-World Model | Mathematical Formula | Generation Speed on Your Mac |
| :--- | :--- | :--- | :--- |
| **1.0 GB** | **Llama 3.2 1B (4-bit)** | $\frac{80\text{ GB/s}}{1.0\text{ GB}}$ | ⚡ **`~80 tokens/sec`** (Instantaneous) |
| **4.68 GB** | **Qwen 2.5 Coder 7B (4-bit)** | $\frac{80\text{ GB/s}}{4.68\text{ GB}}$ | 🚀 **`17.05 tokens/sec`** *(Empirically Measured)* |
| **16.0 GB** | **Qwen 2.5 Coder 32B (4-bit)** | $\frac{80\text{ GB/s}}{16.0\text{ GB}}$ | ⏳ **`~5.0 tokens/sec`** (Human Reading Speed) |
| **140.0 GB** | **Llama 3.1 70B (FP16 unquantized)**| $\frac{80\text{ GB/s}}{140.0\text{ GB}}$ | 🐢 **`~0.57 tokens/sec`** (1 word every 2 seconds) |

> [!IMPORTANT]
> **The Golden Law of Quantization:**  
> Quantizing a model from 16-bit to 4-bit shrinks file size by $4\times$, which makes it run **$4\times$ faster in silicon** because the memory bus has $4\times$ fewer bytes to move per token!

---

## 5. Global Silicon Memory Hierarchy Matrix

| Hardware Tier | Memory Technology | Bus Width | Peak Bandwidth | Sustained Real-World Bandwidth | Qwen 7B (4.68 GB) Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Raspberry Pi 5 (8 GB)** | LPDDR4X | 32-bit | $17\text{ GB/s}$ | $\approx 15\text{ GB/s}$ | **`~3 to 4 t/s`** |
| **Base Mac (M1/M2/M3/M4)** | Unified LPDDR5 | **128-bit** | $100\text{ to }120\text{ GB/s}$ | ⚡ **$\approx 80\text{ GB/s}$** *(Your Mac)* | **`17.05 t/s`** |
| **Mac Pro (M-Pro)** | Unified LPDDR5 | **256-bit** | $150\text{ to }200\text{ GB/s}$ | 🚀 **$\approx 140\text{ GB/s}$** | **`~30 t/s`** |
| **Mac Max (M-Max)** | Unified LPDDR5 | **512-bit** | $300\text{ to }400\text{ GB/s}$ | 🏎️ **$\approx 320\text{ GB/s}$** | **`~68 t/s`** |
| **Mac Ultra (M-Ultra)** | Unified LPDDR5 | **1024-bit** | $800\text{ GB/s}$ | 🚀 **$\approx 650\text{ GB/s}$** | **`~140 t/s`** |
| **Office RTX 3090 (24 GB)** | GDDR6X | **384-bit** | $936\text{ GB/s}$ | 💥 **$\approx 820\text{ GB/s}$** | **`~120+ t/s`** |
| **NVIDIA H100 (80 GB)** | HBM3 (Stacked) | **5,120-bit** | **$3,350\text{ GB/s}$** | 🌌 **$\approx 2,800\text{ GB/s}$** | **`~350+ t/s`** |

---

## 6. Anonymous Heap vs. File-Backed `mmap` Page Eviction

Why doesn't macOS write `mmap` GGUF models into swap space under memory pressure?

```
A. ANONYMOUS HEAP (malloc / new):
   - Created in RAM out of thin air; does NOT exist on SSD.
   - Under memory pressure: The kernel CANNOT delete it.
   - Action: Kernel MUST pause and WRITE those dirty bytes to `/swapfile` (Heavy SSD write I/O).

B. FILE-BACKED MMAP (PROT_READ on GGUF):
   - The 4.68 GB file ALREADY lives on the NVMe SSD.
   - The RAM pages were NEVER modified (Read-Only).
   - Under memory pressure: Kernel does NOT write to swap.
   - Action: Kernel instantly DISCARDS the physical RAM frames (Zero SSD write overhead!).
```

### What happens if another app steals RAM? (Disk Thrashing)
1. If Chrome steals all RAM, macOS discards the model's physical RAM pages.
2. When `llama_decode()` accesses the next layer, the MMU encounters a **Major Page Fault**!
3. The kernel pauses the CPU/GPU thread and reads the 4 KB page back from the NVMe SSD into RAM.
4. **The Symptom:** Generation speed collapses from **`17 tokens/sec` down to `0.5 tokens/sec`** because reading from SSD is $100\times$ slower than RAM.
5. **The Fix (`mlock`):** Calling POSIX `mlock(addr, size)` pins the 4.68 GB of physical RAM frames to hardware, preventing the OS from evicting them under any circumstances.

---

## 7. The 7-Step C++ Inference Pipeline (`llama.h` API)

From our verified C++ driver ([`custom_infer.cpp`](../learn_projects/custom_inference/custom_infer.cpp)):

```cpp
// 1. Dynamic Linker & Shaders
ggml_backend_load_all(); // Loads Metal GPU Shaders

// 2. Model Mapping (mmap)
llama_model_params mparams = llama_model_default_params();
mparams.n_gpu_layers = 99;
llama_model * model = llama_model_load_from_file("qwen7b.gguf", mparams); // 624 ms

// 3. Tokenization (BPE String -> Int IDs)
// "Explain virtual memory..." -> [840, 20772, 4108, 4938, 304, 220, 16, 11652, 13]
llama_tokenize(vocab, prompt, ..., prompt_tokens.data(), ...);

// 4. Context & KV-Cache Allocation
llama_context_params cparams = llama_context_default_params();
cparams.n_ctx = 73; // 14.00 MiB RAM allocated (7 MB Key, 7 MB Value)
llama_context * ctx = llama_init_from_model(model, cparams);

// 5. Sampler Initialization (Greedy)
llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

// 6. The Prefill Phase (GEMM Parallel Prompt Evaluation)
llama_batch batch = llama_batch_get_one(prompt_tokens.data(), 9);
llama_decode(ctx, batch); // 3.73 ms TTFT!

// 7. The Autoregressive Generation Loop (GEMV Sequential Token-by-Token)
for (int i = 0; i < n_predict; ++i) {
    llama_token id = llama_sampler_sample(smpl, ctx, -1);
    if (llama_vocab_is_eog(vocab, id)) break;
    print_piece(vocab, id); // " Virtual", " memory", etc.
    batch = llama_batch_get_one(&id, 1);
    llama_decode(ctx, batch); // 17.05 tokens/sec using KV-Cache!
}

// 8. Deterministic Memory Teardown
llama_sampler_free(smpl);
llama_free(ctx);         // Frees 14 MB KV-Cache
llama_model_free(model); // munmap() releases 4.68 GB model
```

---

## 8. OS Concepts Repurposed for AI Engines

* **Pages & Page Tables $\to$ PagedAttention (vLLM):** Divides KV-cache into discrete pages to eliminate 60–80% GPU memory fragmentation.
* **Free-Block Queues:** Manages available KV-cache blocks in GPU VRAM in $O(1)$ time.
* **Distributed Rank & World Size:** Splits 70B+ models across multi-GPU nodes using NCCL over NVLink (`world_size = total GPUs`, `rank = current GPU ID`).


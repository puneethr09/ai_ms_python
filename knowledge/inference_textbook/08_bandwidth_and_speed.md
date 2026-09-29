# Chapter 8: Bandwidth and Speed, Why Decode Is Slow

> [← Chapter 7](07_kv_cache.md) · [Contents](README.md) · [Chapter 9 →](09_safetensors_file_format.md)

**Reading time:** about 15 minutes. **Status:** ✅ covered. This is the most important chapter for the job.

## In one minute

- A GPU (or CPU) is two things: **memory chips** (the warehouse, big and far) and **compute cores** (the desk, fast but holds only a few MB).
- The 988 MB of weights can't live on the desk, so **for every word, every weight travels through the pipe**. On this M3: 988 MB ÷ 92 GB/s = **10.7 ms → at most ~93 words/sec**.
- **Decode is bandwidth-bound**: the cores finish one row's math almost instantly, then sit idle waiting for the next weights. **Prefill is compute-bound**: each weight fetched is used for many rows.
- In decode, the cores are idle **most of the time**. Using that idle time is what inference engineering is about: **batching, quantization, speculative decoding**.

---

## 1. "The model is loaded in the GPU": the warehouse and the desk

A common belief: *once the model is loaded into the GPU, it's all readily available, and only the small row has to move around.* The catch is that a GPU is **two separate things**:

```
┌────────────────────┐          ┌──────────────────┐
│ MEMORY CHIPS       │  pipe    │ COMPUTE CORES    │
│ (the warehouse)    │ ══════►  │ (the desk)       │
│ 16 GB on the M3    │ 92 GB/s  │ does the math    │
│ all 988 MB of      │          │ holds only a few │
│ weights live here  │          │ MB at a time     │
└────────────────────┘          └──────────────────┘
```

"Loaded into the GPU" means the weights are in the **warehouse**. A core can only multiply numbers on its **desk**, which holds a few MB. The model is 988 MB, so it can't stay on the desk. It's the latency pyramid from chapter 01 of the systems book: desk versus a 4-minute walk to the warehouse.

So for **every generated word**:

```
layer 1  (30 MB)  warehouse → desk → multiply → evicted
layer 2  (30 MB)  warehouse → desk → multiply → evicted
...
layer 24 (30 MB)
dictionary (272 MB) warehouse → desk → output match
───────────────────────────────────────────
988 MB through a 92 GB/s pipe = 10.7 ms
→ at most ~93 words/sec
```

Meanwhile, what else moves?

```
per word:  the row  ≈ 0.004 MB   (896 numbers)
           weights  =   988 MB
```

**Moving the row is free; moving the weights is the cost.**

Other corrections:
- **Output matching happens on the GPU**, not the CPU: it's just one more matrix multiply (row × dictionary table). In Phase 1's numpy, everything runs on the CPU.
- On an M3, the CPU and GPU share the **same memory chips** ("unified memory"). Both hit the same ~90 GB/s pipe.
- The dictionary sits in the warehouse like everything else: **272 MB, 28% of each word's traffic.**

## 2. The ceiling formula

```
max words/sec  =  bandwidth  ÷  bytes read per word
```

Measured in Phase 0 on this M3 (theoretical peak 102.4 GB/s):

| Device | Measured | % of theoretical |
| :--- | ---: | ---: |
| CPU, 8 threads | 92.0 GB/s | 90% |
| GPU (MPS) | 87.1 GB/s | 85% |
| CPU, 1 thread | 64 GB/s | 63% |

One thread can't saturate the pipe, so the Phase 2 C engine needs threads.

```
Qwen2.5-0.5B, BF16:  92 / 0.988  ≈ 93 words/sec
```

### Counting passes (a common off-by-one)

Every trip (pass) produces **exactly one** word, and every pass reads all the weights once. Prefill's pass produces word 1.

```
10-word answer = 10 passes (1 prefill + 9 decode)
               = 10 × 988 MB ≈ 9.9 GB
               ÷ 92 GB/s     ≈ 107 ms
```

(Your first attempt counted 11 passes: prefill + 10 decode. In a generate loop, that kind of off-by-one becomes a real bug.)

## 3. Compute-bound vs bandwidth-bound

| Limit | Question it answers |
| :--- | :--- |
| **Compute** | How fast can the cores multiply? |
| **Memory bandwidth** | How fast can the weights reach the cores? |

**Decode:** each weight fetched is used for about **1 multiply-add** (one row). The cores finish instantly and wait. → **Bandwidth-bound.**

**Prefill:** each weight fetched is used for **every prompt row**. The more rows, the more math per byte fetched, until the cores become the limit. → **Compute-bound.**

> **Decode is bandwidth-bound; prefill is compute-bound.** That sentence is the core of inference engineering.

### How idle are the cores in decode? (rough numbers)

- Decode work: about 2 operations per weight (multiply + add) ≈ **1 GFLOP per word**.
- At the ~93 words/sec ceiling, that's about **93 GFLOP/s** of useful math.
- The M3's GPU can do very roughly **3.5 TFLOP/s** (an estimate; Apple doesn't publish an exact figure).

So in single-user decode the GPU is doing useful math only about **3% of the time**. Put another way: the GPU can do roughly **40 operations in the time it takes to fetch one byte**, but decode only has about **1 operation per byte** to give it. (This ratio is **arithmetic intensity**; the **Roofline model** in the reading list formalizes it.)

## 4. Using the idle cores: the three big tricks

Your own observation: *"in decode a lot of GPU resources are wasted compared to prefill, and that's where better inferencing comes into play."* Exactly.

| Trick | Idea | Why it works | Phase |
| :--- | :--- | :--- | :--- |
| **Batching** | Serve many users at once | Weights are read **once** and used for N users' rows. Pipe cost stays about the same; output grows ~N×. Idle cores absorb the extra math. Each user still has their own context and KV cache. | 6 |
| **Quantization** | Store weights in 8 or 4 bits instead of 16 | Fewer bytes per weight → the pipe finishes sooner. 4-bit ≈ 4× less traffic. | 3 |
| **Speculative decoding** | A small model guesses several words; the big model checks them all in one pass | Checking k guesses is like a tiny prefill: one weight read, k rows. | 6 |

And two smaller ones you've met:
- **Only run the output match on the last row in prefill** (the other rows' predictions are thrown away). Saves 151,936 × 896 multiply-adds per skipped row.
- **The KV cache** ([Chapter 7](07_kv_cache.md)) avoids recomputing the prompt, but its own size must also be read every step, which is why GQA exists.

## 5. How model shape affects speed

- **More layers = more bytes per word = slower.** 100 layers at this width ≈ 3.25 GB per word ≈ 28 words/sec.
- **Every layer costs the same** (~30 MB), because the model is a tube, not a cone ([Chapter 6](06_layers_and_the_forward_pass.md)). So time per word ≈ 24 equal steps + the dictionary step.
- **A 7B model** is 14 GB in BF16 → at most ~6.5 words/sec on this M3. At 4 bits (~4 GB) → ~23 words/sec.

## 6. Where your Phase 1 engine will land

- Your loader converts BF16 → **FP32**, doubling the model to about **2 GB**. That alone halves the ceiling to ~46 words/sec.
- No KV cache: every trip recomputes the whole sequence.
- numpy overhead, single-threaded parts, no fused kernels.

Result: about **1 word/sec** versus a ceiling of 93. **Phase 1 reflection (write it before Phase 2):** in your own words, where does the missing 99% go?

---

## Check yourself

1. "The model is already in GPU memory, so there's no bandwidth problem." What's wrong with that?
2. A 10-word answer: how many passes, how many GB moved, roughly how long on this M3?
3. Why is prefill compute-bound but decode bandwidth-bound?
4. Name one trick that uses decode's idle compute, and why it works.
5. In decode, does the 896-number row travel through the pipe? Why does or doesn't it matter?

<details>
<summary>Answers</summary>

1. The weights sit in the memory chips (warehouse). The cores (desk) hold only a few MB, so all 988 MB must stream through the pipe for every word.
2. 10 passes (1 prefill + 9 decode), ~9.9 GB, ~107 ms.
3. In prefill, each weight fetched is used for every prompt row (lots of math per byte). In decode, each weight is used for one row (≈1 operation per byte), so the cores wait on memory.
4. Batching: one weight read serves N users' rows, so output grows ~N× for about the same memory traffic. (Or quantization: fewer bytes to move. Or speculative decoding: check several guessed words per weight read.)
5. Effectively no: it's ~2–4 KB and stays on the chip. It doesn't matter next to 988 MB of weights.

</details>

---

> [← Chapter 7](07_kv_cache.md) · [Contents](README.md) · [Chapter 9 →](09_safetensors_file_format.md)

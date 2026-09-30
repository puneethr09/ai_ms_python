# The Inference Textbook

> **How a language model turns text into the next word, and why that is slow.** Built from first principles, with real numbers from Qwen2.5-0.5B and this M3 Mac.

This is the theory half of the [Inference From Scratch](../../learn_projects/inference_from_scratch/) track and the [9-phase roadmap](../06_inference_engineering_roadmap.md). It grows with every lesson. Every number in it was measured on the real model file or computed by hand.

---

## How to read it

- **Order matters.** Each chapter builds on the one before. First time through, read 1 → 9 in order.
- **On the go:** each chapter opens with **"In one minute"**. Read just those for a 10-minute refresher of everything.
- **Pen and paper** for chapters 4 and 5: the toy examples are meant to be computed by hand.
- **Test yourself:** every chapter ends with check questions (answers are folded; tap to reveal). The [flashcards](self_test.md) collect all of them.
- **Status markers:** ✅ covered · 🟡 in progress · ⏸ skipped for now · ⬜ not started.

---

## Contents

| # | Chapter | Time | Status |
| :--- | :--- | ---: | :--- |
| 1 | [What is inference?](01_what_is_inference.md): the loop, prefill vs decode, the 93 words/sec wall | 10 min | ✅ |
| 2 | [Tokens, embeddings and the output match](02_tokens_and_embeddings.md): IDs, 896-number rows, 151,936 logits, weight tying | 15 min | ✅ |
| 3 | [Weights, wires and matrix multiplication](03_weights_and_matrix_multiply.md): row × W, the cookie cutter, training, "0.5B" | 15 min | ✅ |
| 4 | [Attention, from zero](04_attention.md): q, k, v, softmax, the mix, Wo, heads, GQA | 25 min | ✅ |
| 5 | [The MLP, the model's reference books](05_mlp.md): detectors, gate/up/down, why facts live here | 20 min | 🟡 |
| 6 | [Layers and the forward pass](06_layers_and_the_forward_pass.md): the classroom, why 24, logit lens, tube not cone | 25 min | ✅ |
| 7 | [The KV cache](07_kv_cache.md): the filing cabinet, 12 KB per token | 12 min | ✅ |
| 8 | [Bandwidth and speed](08_bandwidth_and_speed.md): warehouse and desk, compute vs bandwidth-bound, batching | 15 min | ✅ |
| 9 | [The model file (safetensors)](09_safetensors_file_format.md): Lesson 1.1, bytes, little-endian, BF16 | 20 min | 🟡 |

**Reference pages**

| Page | Use it for |
| :--- | :--- |
| [Self-test flashcards](self_test.md) | 43 questions with hidden answers, grouped by chapter |
| [Corrections log](corrections.md) | 27 things I believed that were wrong, and the fix. Read before interviews. |
| [Glossary](glossary.md) | Every term in one line, plus Qwen2.5-0.5B's numbers |
| [Learning journal](learning_journal.md) | Honest assessment, topic status, open items |

---

## The whole book on one screen

```
"The capital of France is"
  → tokenizer → [785, 6722, 315, 9625, 374]
  → embedding lookup → 5 rows × 896
  → 24 layers, in order, each:
       attention: rows read earlier rows  (+ add)
       MLP: each row looks up knowledge   (+ add)
  → last row × embedding table → 151,936 logits
  → pick " Paris" → append → repeat

Cost per word: read all 988 MB of weights (+ KV cache)
Ceiling: 92 GB/s ÷ 0.988 GB ≈ 93 words/sec
Decode: bandwidth-bound. Prefill: compute-bound.
```

## Where this sits in the roadmap

```
Phase 0 ✅ Instruments: measured 92 GB/s
Phase 1 🟡 Transformer in numpy     ← this book, so far
Phase 2 ⬜ C engine + KV cache
Phase 3 ⬜ Quantization
Phase 4 ⬜ Metal / MLX
Phase 5 ⬜ CUDA (Colab)
Phase 6 ⬜ Serving (batching, paged KV)
Phase 7 ⬜ Scale (rented GPUs)
Phase 8 ⬜ Upstream PRs
```

New chapters are added as each lesson is completed: tokenizer (1.2), RMSNorm and SwiGLU (1.3), RoPE (1.4), attention in code (1.5), the full model (1.6), sampling (1.7), and then one or more per phase.

## Reproduce the numbers

From `learn_projects/inference_from_scratch/`:

```bash
python -m bench.membw                 # bandwidth (Chapter 8)
python scripts/logit_lens.py          # " Paris" layer by layer (Chapter 6)
xxd -l 64 models/qwen2.5-0.5b/model.safetensors   # the header (Chapter 9)
```

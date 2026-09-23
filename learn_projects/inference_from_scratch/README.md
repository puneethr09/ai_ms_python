# Inference From Scratch

The hands-on track for the [9-phase roadmap](../../knowledge/06_inference_engineering_roadmap.md). Each phase is **built by hand**, **tested against a reference** (PyTorch / Hugging Face) and **measured against the hardware's limit**.

| Phase | Status | Exit criterion |
| :--- | :--- | :--- |
| 0. Instruments | ✅ Done | Real bandwidth measured → [`results/hardware.json`](results/hardware.json) |
| 1. Transformer in numpy | ⏳ In progress | `pytest` all green |
| 2. C engine + KV cache | — | Decode ≥ 70% of `bandwidth ÷ model bytes` |
| 3. Quantization | — | Speed vs perplexity table |
| 4. Metal / MLX | — | GPU decode as % of bandwidth |
| 5. CUDA (Colab) | — | GEMM ≥ 70% of cuBLAS |
| 6. Serving | — | TTFT / TPOT / throughput curves |
| 7. Scale (rented GPU) | — | Profiled multi-GPU run |
| 8. Upstream | — | Merged PRs |

## Setup

```bash
cd learn_projects/inference_from_scratch
pip install -r requirements.txt
python scripts/download_model.py     # Qwen2.5-0.5B, ~1 GB, into models/ (gitignored)
python scripts/make_reference.py     # HF fp32 ground truth -> fixtures/reference.npz (gitignored)
```

## Phase 0: Instruments ✅

```bash
python -m bench.membw
```

Apple M3 (16 GB, 128-bit LPDDR5-6400, 102.4 GB/s theoretical):

| Device | Measured peak | % of theoretical |
| :--- | :--- | :--- |
| CPU (8 threads, read) | **92.0 GB/s** | 90% |
| GPU (MPS copy) | **87.1 GB/s** | 85% |

One thread alone reaches 64 GB/s. A single core can't saturate the bus, so a CPU decode engine needs threads (Phase 2).

[`bench/harness.py`](bench/harness.py) turns later results into percentages of these peaks:

```python
from bench.harness import time_fn, pct_of_bandwidth, predicted_decode_tok_s
predicted_decode_tok_s(988e6)   # Qwen2.5-0.5B in BF16 -> ~93 tok/s ceiling on this M3
```

## Phase 1: A transformer in numpy

Write the code in [`nanoinfer/`](nanoinfer/). Every stub raises `NotImplementedError`, and the tests are your to-do list. Work through them in order:

```bash
pytest tests/test_1_1_safetensors.py   # then 1_2, 1_3, 1_4, 1_5
pytest                                 # all 14 green = Phase 1 done
python -m nanoinfer.generate "The capital of France is" --max-tokens 20
```

**Rule:** `nanoinfer/` may import only `numpy`, `regex` and the standard library. torch and transformers exist only to check your answers.

| Step | File | What to build | Hints |
| :--- | :--- | :--- | :--- |
| 1.1 | `safetensors.py` | Parse the file by hand | 8-byte little-endian header length, then a JSON header, then raw bytes. BF16 is the top 16 bits of an FP32 |
| 1.2 | `tokenizer.py` | Byte-level BPE | NFC normalize → split with the regex in `tokenizer.json` (needs the `regex` module for `\p{L}`) → map bytes to printable unicode (GPT-2 trick) → apply merges by rank |
| 1.3 | `model.py` | `rms_norm`, `swiglu_mlp` | Weights are `(out_features, in_features)` |
| 1.4 | `model.py` | `rope_tables`, `apply_rope` | Qwen rotates the **first half with the second half**, not adjacent pairs |
| 1.5 | `model.py` | Causal GQA `attention` | 14 query heads share 2 KV heads (7 query heads per KV head) |
| 1.6 | `model.py` | `Qwen2.forward` | q/k/v projections **have biases**; `lm_head` is **tied** to `embed_tokens` |
| 1.7 | `sampling.py` | temperature, top-k, top-p | Top-p keeps the smallest set whose probability reaches p |

When `test_hidden_states_layer_by_layer` fails, it names the first layer that diverges, so you know which step to debug.

**Phase 1 reflection:** with no KV cache, generation recomputes the whole sequence for every token (about 1 tok/s). The bandwidth ceiling is about 93 tok/s. Before starting Phase 2, write down in your own words where the missing 99% goes.

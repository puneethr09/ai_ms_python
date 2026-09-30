# Chapter 9: The Model File (safetensors), Lesson 1.1

> [← Chapter 8](08_bandwidth_and_speed.md) · [Contents](README.md) · [Self-test →](self_test.md)

**Reading time:** about 20 minutes. **Status:** 🟡 in progress. Exercise A is done; the tools class, Exercise B and Exercise C (your code) are next.

## Where we are on the map

```
Phase 0 ✅ measured the pipe: 92 GB/s
Phase 1: build the transformer in numpy
  ▶ 1.1 read the weights file   ← here
    1.2 tokenizer (text → IDs)
    1.3 rms_norm, MLP
    1.4 RoPE (positions)
    1.5 attention
    1.6 full forward pass
    1.7 sampling (scores → word)
```

Everything in chapters 2–7 (Wq, Wk, Wv, Wo, gate, up, down, the dictionary) is just numbers in a file on disk. Before any math can run, they must be in memory. Step 1.1 is the doorway: **turn a file into numpy arrays.**

## In one minute

- `model.safetensors` = **8 bytes** (header length N, little-endian) + **N bytes of JSON** (table of contents) + **raw numbers**.
- For your file: N = **32,280**, so the data starts at byte **32,288**. 290 tensors, all BF16.
- **Little-endian** means reverse the **bytes**, not the hex digits.
- **BF16 is the top 2 bytes of an FP32.** To convert, put two zero bytes underneath.

---

## 0. The model folder: which file is which

The download gives you a folder, not one file. The files are listed here in the order a word passes through them:

```
models/qwen2.5-0.5b/
├── config.json              681 B   the model's shape: the "blueprint"
├── generation_config.json   138 B   default settings for the generation loop
├── tokenizer.json           7.0 MB  text ↔ token IDs: the full rulebook
├── vocab.json               2.8 MB  ┐ the same tokenizer, split into two
├── merges.txt               1.7 MB  ┘ older-style files (redundant)
├── tokenizer_config.json    7.2 KB  special tokens + the chat template
├── model.safetensors        988 MB  the 494M weights: the only big file
└── .cache/                          download bookkeeping (ignore)
```

| File | What it is | Used in step |
| :--- | :--- | :--- |
| `config.json` | The shape numbers: `hidden_size: 896`, `num_hidden_layers: 24`, `num_attention_heads: 14`, `num_key_value_heads: 2`, `intermediate_size: 4864`, `rope_theta: 1000000`, `rms_norm_eps: 1e-06`, `tie_word_embeddings: true`. It's plain text and contains **no weights**. | 1.6 |
| `model.safetensors` | The weights: 8-byte length, JSON header, raw BF16 numbers (this chapter) | **1.1** |
| `tokenizer.json` | 151,643 regular tokens, 151,387 merge rules, 22 special tokens, and the regex that splits text ([Chapter 2](02_tokens_and_embeddings.md)) | 1.2 |
| `vocab.json` + `merges.txt` | The same tokenizer in the older GPT-2 format. You only need one of them. | — |
| `tokenizer_config.json` | Which IDs are special (end-of-text = 151643), and the **chat template** (`<\|im_start\|>user…`) | Later (chat, serving) |
| `generation_config.json` | Loop defaults: `do_sample: false` (always pick the top word), stop at ID 151643, at most 2048 new tokens | 1.7 |

**How to tell them apart at a glance:**
- **Size.** The weights are always the huge one.
- **Extension.** `.json` and `.txt` are text: open them in the editor. `.safetensors` is binary, so you need code or `xxd` to read it (`head -c 8 model.safetensors | xxd`).

**Spare dictionary rows:** `config.json` says `vocab_size: 151936`, but the tokenizer has only 151,643 + 22 = 151,665 tokens. That leaves 271 rows that no text can ever produce. 151,936 = 1,187 × 128: the row count is a multiple of 128, which suits GPU matrix shapes, and it leaves room to add special tokens later without changing the tensor shapes.

> **Lesson from Exercise B:** the first attempt opened `config.json` instead of `model.safetensors`. Python didn't complain: it turned the text `{\n  "arc` into the integer 7,165,896,756,295,633,531. Code that reads bytes never errors on the wrong file. **Always check against a number you already know** (`assert n == 32280`).

## 1. The file's layout

```
┌──────────┬────────────────────┬──────────────────┐
│ 8 bytes  │ N bytes of JSON    │ raw numbers,     │
│ = N      │ = table of contents│ back to back     │
└──────────┴────────────────────┴──────────────────┘
```

A table-of-contents entry:

```json
"model.layers.0.self_attn.k_proj.weight": {
  "dtype": "BF16",
  "shape": [128, 896],
  "data_offsets": [start, end]
}
```

That's Wk for layer 0: 896 → 128, as in [Chapter 4](04_attention.md). Offsets are counted **from the start of the data section** (byte 8 + N), not from the start of the file. There's also a `__metadata__` key (`{"format": "pt"}`) that is not a tensor.

Facts about your real file:

| Fact | Value |
| :--- | ---: |
| First 8 bytes | `18 7e 00 00 00 00 00 00` |
| N (header length) | 32,280 |
| Data starts at | 32,288 |
| Tensors | 290 (24 layers × 12 + embedding + final norm) |
| dtype | all BF16 |
| Data bytes | 988,065,536 |
| File size = 8 + N + data | 988,097,824 ✅ |
| `lm_head.weight` present? | No: tied to the embedding ([Chapter 2](02_tokens_and_embeddings.md)) |

## 2. Little-endian: the lesson from Exercise A

Your first answer was N = `e781` = 59,265. The first 8 bytes are `18 7e 00 00 00 00 00 00`. You reversed the **hex digits** (`187e` → `e781`). Little-endian means reversing the **bytes**, and each byte's two hex digits stay together:

```
bytes in file:  [18] [7e] [00] [00] [00] [00] [00] [00]
reversed:       [00] [00] [00] [00] [00] [00] [7e] [18]
value:          0x7e18 = 7×4096 + 14×256 + 1×16 + 8
              = 32,280
```

**Why it matters:** the smallest addressable unit of memory is a byte. Little-endian puts the least significant **byte** at the lowest address. Get it wrong in a loader and every offset shifts: you read garbage that still looks like valid floats. That's the worst kind of bug, because nothing crashes.

Practice (you answered this correctly): `01 02 00 00` → reversed `00 00 02 01` → 0x0201 = 2×256 + 1 = **513**.

## 3. A toy file, byte by byte

One tensor, `x = [1.0, 2.0]`, in BF16:

```
bytes 0–7:   37 00 00 00 00 00 00 00
             → N = 0x37 = 55
bytes 8–62:  {"x":{"dtype":"BF16","shape":[2],
              "data_offsets":[0,4]}}   (55 bytes)
bytes 63–66: 80 3F 00 40
             → 0x3F80 = 1.0, 0x4000 = 2.0
```

Reading any safetensors file:
1. Read 8 bytes → decode as a little-endian unsigned 64-bit integer → N.
2. Read N bytes → parse as JSON → the table of contents.
3. For each tensor: take bytes `[start, end)` counted from byte 8 + N, turn them into numbers, give them the right shape.

## 4. BF16: the top half of an FP32

An FP32 is 4 bytes: 1 sign bit, 8 exponent bits, 23 fraction bits. **BF16 keeps the first 16 bits** (sign, all 8 exponent bits, 7 fraction bits) and drops the rest. Same range as FP32, less precision.

```
1.0 as FP32 = 0x3F80_0000
1.0 as BF16 = 0x3F80        (top half)
2.0 as FP32 = 0x4000_0000 → BF16 0x4000
```

To go back: **glue two zero bytes underneath** (shift left by 16 bits). numpy has no bfloat16 type, so you'll do this with integer arrays and a `.view`. That's the tools class.

## 5. Why this format, and why it matters for inference

- **Safety:** the older `.bin`/`.pt` format is a Python **pickle**, and loading one can run arbitrary code: a downloaded model could run code on your machine. safetensors is pure data. That's why it became the standard.
- **Zero-copy:** because the numbers are raw and packed, a real engine can `mmap` the file and use the tensors in place, without copying (chapters 02 and 05 of the systems book). llama.cpp and vLLM do this; llama.cpp loads a 4.68 GB model in 624 ms this way.
- **Two different 988 MB costs:**
  - **Loading:** once, from SSD at ~3 GB/s ≈ 0.3 s.
  - **Decoding:** 988 MB from RAM **for every word** (10.7 ms each). The loader is paid once; the forward pass is paid on every token.
- **FP32 doubles memory:** after your conversion the model takes ~2 GB in RAM and ~2 GB of traffic per word, halving the ceiling. Fine for Phase 1 (correctness first); a problem for Phase 2 (speed), where you'll keep BF16 or quantize.

## 6. The tools (for Exercise B)

Try each in a Python REPL before combining them.

**Tool 0: find the file, and close it automatically**

A relative path like `"learn_projects/..."` is resolved from the terminal's **current folder**, not from the script's folder, so it breaks when you run the script from somewhere else. Build the path from the script's own location instead:

```python
from pathlib import Path

HERE = Path(__file__).resolve().parent     # the folder this script is in
MODEL = HERE.parent / "models" / "qwen2.5-0.5b" / "model.safetensors"

with open(MODEL, "rb") as f:               # "rb" = read raw bytes
    first = f.read(8)
# the file is closed here, even if something crashed inside
```

**Tool 1: read raw bytes**
```python
f = open("models/qwen2.5-0.5b/model.safetensors", "rb")
first = f.read(8)      # bytes object
first.hex(" ")         # '18 7e 00 00 00 00 00 00'
f.tell()               # 8: the next read continues here
```

**Tool 2: bytes → numbers with `struct`**

| Char | Meaning |
| :--- | :--- |
| `<` / `>` | little-endian / big-endian |
| `H` / `I` / `Q` | unsigned 16 / 32 / 64-bit integer |
| `f` | 32-bit float |

```python
import struct
struct.unpack("<I", bytes([1, 2, 0, 0]))  # (513,)
struct.unpack(">I", bytes([1, 2, 0, 0]))  # (16908288,)
```

**Tool 3: JSON from bytes**
```python
import json
h = json.loads(b'{"a": {"dtype": "BF16", "shape": [2, 3]}}')
h["a"]["shape"]              # [2, 3]
```

**Tool 4: summarise**
```python
import math
math.prod([896, 151936])     # 136134656
{v["dtype"] for v in h.values()}   # unique dtypes
```

Still to teach before Exercise C: `np.frombuffer`, integer dtypes (`uint16`, `uint32`), shifting, and `.view` to reinterpret bits.

## 7. Exercises

**Exercise A ✅ done:** decode the first 8 bytes by hand (`xxd -l 64 models/qwen2.5-0.5b/model.safetensors`).

**Exercise B (next):** in a scratch file `explore_header.py`, step by step:
1. Read 8 bytes, unpack N. Expect **32280**.
2. `f.read(N)`, `json.loads`, print `len(header)`. (Watch out for `__metadata__`.)
3. Print dtype, shape and offsets of layer 0's `q_proj.weight` and `k_proj.weight`. Why is k smaller? (Answer: GQA, [Chapter 4](04_attention.md).)
4. Is `"lm_head.weight"` there? Check `tie_word_embeddings` in `config.json`.
5. Sum `end - begin` over all tensors; compare `8 + N + total` with `os.path.getsize(path)`.

**Exercise C:** implement `load()` in [`nanoinfer/safetensors.py`](../../learn_projects/inference_from_scratch/nanoinfer/safetensors.py) (BF16 → FP32) until this passes:
```bash
pytest tests/test_1_1_safetensors.py
```

---

## Check yourself

1. The first 8 bytes are `10 00 00 00 00 00 00 00`. How long is the JSON header?
2. In that file, a tensor has `data_offsets: [0, 8]`. At which byte of the file does its data start, and how many BF16 numbers does it hold?
3. BF16 `0x4000` is 2.0. What is it as an FP32, in hex?
4. Why do we convert to FP32 in Phase 1, and what does it cost in decode speed?

<details>
<summary>Answers</summary>

1. 0x10 = 16 bytes.
2. Byte 8 + 16 + 0 = 24. 8 bytes ÷ 2 bytes each = 4 numbers.
3. 0x40000000.
4. numpy has no bfloat16, and FP32 matches the reference tests. It doubles the bytes per word (~2 GB), halving the bandwidth ceiling (~46 words/sec instead of ~93).

</details>

---

> [← Chapter 8](08_bandwidth_and_speed.md) · [Contents](README.md) · [Self-test →](self_test.md)
